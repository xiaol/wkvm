"""Native Gemma token-id and opt-in OpenAI chat HTTP endpoint.

The canonical wkvm endpoint is token-id `/v1/stream`, including explicit
stateful sessions that accept an initial prompt or later token deltas.
`/v1/completions` exposes the same engine through the OpenAI completions
streaming shape used by vLLM and SGLang benchmarks, limited to single-prompt
greedy token-id requests.
`--enable-openai-chat` adds tokenizer-backed `/v1/chat/completions` support.
"""

from __future__ import annotations

import hashlib
import json
import math
import signal
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Iterator
from urllib.parse import unquote, urlparse


DEFAULT_MAX_REQUEST_BODY_BYTES = 8 * 1024 * 1024
DEFAULT_REQUEST_READ_TIMEOUT_S = 30.0
_REQUEST_BODY_READ_CHUNK_BYTES = 64 * 1024
PARENT_TOKEN_CHAT_CONTRACT = "parent-token-v1"
_MAX_CHAT_IDENTITY_BYTES = 512


@dataclass
class _ChatTurn:
    session_id: str
    response_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int
    break_mask: list[bool] | None
    deadline: float | None
    session_kind: str
    input_mode: str
    forced_output_token_ids: list[int] | None = None
    request: Any | None = None
    engine_req_id: str | None = None
    state: str = "pending"
    output_token_ids: list[int] = field(default_factory=list)
    finish_reason: str | None = None
    error: str | None = None
    metrics: dict[str, Any] | None = None
    cancel_requested: bool = False
    cuda_cache_emptied: bool = False
    session_reused: bool = False
    candidate_output_token_ids: list[int] = field(default_factory=list)
    teacher_forcing_overwrite_s: float = 0.0
    teacher_forcing_overwrites: int = 0
    chat_messages: list[dict[str, str]] | None = None
    chat_codec: Any | None = None
    parent_bound_contract: str | None = None
    assistant_message_id: str | None = None
    user_message_id: str | None = None
    parent_message_id: str | None = None
    session_reuse_mode: str = "new_session"
    parent_bound_continuation_attempted: bool = False
    parent_bound_continuation_accepted: bool = False
    parent_bound_continuation_rejection_reason: str | None = None

    @property
    def finished(self) -> bool:
        return self.state == "finished"


@dataclass
class _ChatSession:
    session_id: str
    engine_req_id: str
    session_kind: str
    request: Any
    active_turn: _ChatTurn | None
    last_access: float
    parent_bound_contract: str | None = None
    canonical_input_messages: list[dict[str, str]] | None = None
    canonical_prompt_token_ids: list[int] | None = None
    visible_output: str | None = None
    visible_parent_history_digest: str | None = None
    retained_token_digest: str | None = None
    last_assistant_message_id: str | None = None
    last_user_message_id: str | None = None
    last_response_id: str | None = None
    generation: int = 0


def _chat_identity(value: str | None, field_name: str) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if len(normalized.encode("utf-8")) > _MAX_CHAT_IDENTITY_BYTES:
        raise ValueError(
            f"{field_name} exceeds {_MAX_CHAT_IDENTITY_BYTES} UTF-8 bytes"
        )
    return normalized


def _token_history_digest(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _message_history_digest(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class BoundedGemmaService:
    def __init__(
        self,
        engine,
        *,
        max_queue: int = 64,
        batch_wait_s: float = 0.01,
        request_timeout_s: float | None = None,
        max_completed_requests: int | None = 4096,
        chat_session_ttl_s: float | None = 1800.0,
        max_chat_sessions: int | None = None,
        cuda_empty_cache: Callable[[], None] | None = None,
        enable_token_session_teacher_forcing: bool = False,
    ) -> None:
        if max_queue < 1:
            raise ValueError("max_queue must be >= 1")
        if request_timeout_s is not None:
            request_timeout_s = float(request_timeout_s)
            if not math.isfinite(request_timeout_s) or request_timeout_s <= 0:
                raise ValueError("request_timeout_s must be finite and > 0 or None")
        if max_completed_requests is not None and max_completed_requests < 1:
            raise ValueError("max_completed_requests must be >= 1 or None")
        if chat_session_ttl_s is not None:
            chat_session_ttl_s = float(chat_session_ttl_s)
            if not math.isfinite(chat_session_ttl_s) or chat_session_ttl_s <= 0:
                raise ValueError("chat_session_ttl_s must be finite and > 0 or None")
        if max_chat_sessions is None:
            arena_slots = getattr(getattr(engine, "arena", None), "num_slots", None)
            if arena_slots is not None:
                max_chat_sessions = int(arena_slots)
        if max_chat_sessions is not None and max_chat_sessions < 1:
            raise ValueError("max_chat_sessions must be >= 1 or None")
        if cuda_empty_cache is not None and not callable(cuda_empty_cache):
            raise TypeError("cuda_empty_cache must be callable or None")
        self.engine = engine
        self.max_queue = max_queue
        self.batch_wait_s = max(0.0, float(batch_wait_s))
        self.request_timeout_s = request_timeout_s
        self.max_completed_requests = max_completed_requests
        self.chat_session_ttl_s = chat_session_ttl_s
        self.max_chat_sessions = max_chat_sessions
        self.cuda_empty_cache = cuda_empty_cache
        self.cuda_empty_cache_calls = 0
        self.enable_token_session_teacher_forcing = bool(
            enable_token_session_teacher_forcing
        )
        self.lock = threading.RLock()
        self.cv = threading.Condition(self.lock)
        self.engine_lock = threading.RLock()
        self.ready = True
        self.closed = False
        self.cancelled: set[str] = set()
        self.total_requests = 0
        self.total_errors = 0
        self.total_cancelled = 0
        self.total_timed_out = 0
        self.last_error: str | None = None
        self._pending: deque[tuple[Any, list[bool] | None]] = deque()
        self._pending_chat: deque[_ChatTurn] = deque()
        self._requests: dict[str, Any] = {}
        self._deadlines: dict[str, float] = {}
        self._completed_order: deque[str] = deque()
        self._chat_sessions: dict[str, _ChatSession] = {}
        self._chat_sessions_by_engine_id: dict[str, _ChatSession] = {}
        self._chat_active_turns: dict[str, _ChatTurn] = {}
        self._chat_req_counter = 0
        self._chat_exact_prefix_reuse_hits = 0
        self._parent_bound_continuation_hits = 0
        self._parent_bound_continuation_misses = 0
        self._parent_bound_continuation_rejections: dict[str, int] = {}
        self._worker = threading.Thread(target=self._run_engine, daemon=True)
        self._worker.start()

    def generate(
        self,
        *,
        prompt_ids: list[int],
        max_tokens: int,
        req_id: str | None = None,
        break_mask: list[bool] | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        from wkvm.core.request import Request

        started = time.perf_counter()
        effective_timeout = self.request_timeout_s if timeout_s is None else float(timeout_s)
        if effective_timeout is not None and (
            not math.isfinite(effective_timeout) or effective_timeout <= 0
        ):
            raise ValueError("timeout_s must be finite and > 0 or None")
        deadline = None if effective_timeout is None else started + effective_timeout
        with self.cv:
            request = self._enqueue_locked(
                Request(
                    prompt_token_ids=list(prompt_ids),
                    max_new_tokens=int(max_tokens),
                    req_id=req_id or f"gemma-http-{self.total_requests}",
                ),
                break_mask=break_mask,
            )
        while not request.status.is_finished:
            timed_out = False
            with self.cv:
                if request.status.is_finished:
                    break
                wait_s = 0.25
                if deadline is not None:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        timed_out = True
                    else:
                        wait_s = min(wait_s, remaining)
                if not timed_out:
                    self.cv.wait(timeout=wait_s)
                    if self.closed:
                        raise RuntimeError("service is closed")
            if timed_out:
                if self._timeout_request(request.req_id):
                    raise TimeoutError(f"request {request.req_id} timed out")
        with self.cv:
            with self.engine_lock:
                trace = self.engine.finished_traces.get(request.req_id)
            payload = {
                "req_id": request.req_id,
                "tokens": list(request.output_token_ids),
                "finish_reason": request.status.name.removeprefix("FINISHED_").lower(),
                "error": None if trace is None else trace.error,
                "latency_s": round(time.perf_counter() - started, 6),
                "metrics": trace.as_dict() if trace is not None else None,
            }
            return payload

    def submit(
        self,
        *,
        prompt_ids: list[int],
        max_tokens: int,
        req_id: str | None = None,
        break_mask: list[bool] | None = None,
        timeout_s: float | None = None,
    ) -> dict[str, Any]:
        from wkvm.core.request import Request

        effective_timeout = self.request_timeout_s if timeout_s is None else float(timeout_s)
        if effective_timeout is not None and (
            not math.isfinite(effective_timeout) or effective_timeout <= 0
        ):
            raise ValueError("timeout_s must be finite and > 0 or None")
        deadline = None if effective_timeout is None else time.perf_counter() + effective_timeout
        with self.cv:
            request = self._enqueue_locked(
                Request(
                    prompt_token_ids=list(prompt_ids),
                    max_new_tokens=int(max_tokens),
                    req_id=req_id or f"gemma-http-{self.total_requests}",
                ),
                break_mask=break_mask,
            )
            if deadline is not None:
                self._deadlines[request.req_id] = deadline
            return {"req_id": request.req_id, "status": request.status.name.lower()}

    def stream(
        self,
        *,
        prompt_ids: list[int],
        max_tokens: int,
        req_id: str,
        break_mask: list[bool] | None = None,
        timeout_s: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        from wkvm.core.request import Request

        started = time.perf_counter()
        effective_timeout = self.request_timeout_s if timeout_s is None else float(timeout_s)
        if effective_timeout is not None and (
            not math.isfinite(effective_timeout) or effective_timeout <= 0
        ):
            raise ValueError("timeout_s must be finite and > 0 or None")
        deadline = None if effective_timeout is None else started + effective_timeout
        with self.cv:
            request = self._enqueue_locked(
                Request(
                    prompt_token_ids=list(prompt_ids),
                    max_new_tokens=int(max_tokens),
                    req_id=req_id,
                ),
                break_mask=break_mask,
            )

        emitted = 0
        yield {"type": "queued", "req_id": request.req_id}
        try:
            while True:
                timed_out = False
                with self.cv:
                    while (
                        len(request.output_token_ids) == emitted
                        and not request.status.is_finished
                    ):
                        wait_s = 0.25
                        if deadline is not None:
                            remaining = deadline - time.perf_counter()
                            if remaining <= 0:
                                timed_out = True
                                break
                            else:
                                wait_s = min(wait_s, remaining)
                        self.cv.wait(timeout=wait_s)
                        if self.closed:
                            raise RuntimeError("service is closed")

                    if timed_out:
                        tokens = []
                        start_index = emitted
                        finished = False
                        finish_reason = None
                        error = None
                        metrics = None
                    else:
                        tokens = request.output_token_ids[emitted:]
                        start_index = emitted
                        emitted = len(request.output_token_ids)
                        finished = request.status.is_finished
                        finish_reason = (
                            request.status.name.removeprefix("FINISHED_").lower()
                            if finished
                            else None
                        )
                        with self.engine_lock:
                            trace = self.engine.finished_traces.get(request.req_id)
                        error = None if trace is None else trace.error
                        metrics = trace.as_dict() if trace is not None else None

                if timed_out:
                    if self._timeout_request(request.req_id):
                        raise TimeoutError(f"request {request.req_id} timed out")
                    continue

                for offset, token in enumerate(tokens):
                    yield {
                        "type": "token",
                        "req_id": request.req_id,
                        "index": start_index + offset,
                        "token": int(token),
                    }
                if finished:
                    yield {
                        "type": "finish",
                        "req_id": request.req_id,
                        "finish_reason": finish_reason,
                        "error": error,
                        "latency_s": round(time.perf_counter() - started, 6),
                        "metrics": metrics,
                    }
                    return
        finally:
            if not request.status.is_finished:
                self._cancel_request(request.req_id)

    def generate_chat(
        self,
        *,
        prompt_ids: list[int],
        max_tokens: int,
        session_id: str,
        req_id: str,
        break_mask: list[bool] | None = None,
        timeout_s: float | None = None,
        chat_messages: list[dict[str, str]] | None = None,
        chat_codec: Any | None = None,
        parent_bound_contract: str | None = None,
        assistant_message_id: str | None = None,
        user_message_id: str | None = None,
        parent_message_id: str | None = None,
    ) -> dict[str, Any]:
        tokens: list[int] = []
        finish: dict[str, Any] | None = None
        for event in self.stream_chat(
            prompt_ids=prompt_ids,
            max_tokens=max_tokens,
            session_id=session_id,
            req_id=req_id,
            break_mask=break_mask,
            timeout_s=timeout_s,
            chat_messages=chat_messages,
            chat_codec=chat_codec,
            parent_bound_contract=parent_bound_contract,
            assistant_message_id=assistant_message_id,
            user_message_id=user_message_id,
            parent_message_id=parent_message_id,
        ):
            if event["type"] == "token":
                tokens.append(int(event["token"]))
            elif event["type"] == "finish":
                finish = event
        if finish is None:
            raise RuntimeError(f"chat request {req_id} ended without a finish event")
        return {
            "req_id": req_id,
            "tokens": tokens,
            "finish_reason": finish.get("finish_reason"),
            "error": finish.get("error"),
            "latency_s": finish.get("latency_s"),
            "metrics": finish.get("metrics"),
        }

    def stream_chat(
        self,
        *,
        prompt_ids: list[int],
        max_tokens: int,
        session_id: str,
        req_id: str,
        break_mask: list[bool] | None = None,
        timeout_s: float | None = None,
        chat_messages: list[dict[str, str]] | None = None,
        chat_codec: Any | None = None,
        parent_bound_contract: str | None = None,
        assistant_message_id: str | None = None,
        user_message_id: str | None = None,
        parent_message_id: str | None = None,
        _session_kind: str = "chat",
        _input_mode: str = "full_prompt",
        _forced_output_ids: list[int] | None = None,
    ) -> Iterator[dict[str, Any]]:
        started = time.perf_counter()
        effective_timeout = self.request_timeout_s if timeout_s is None else float(timeout_s)
        if effective_timeout is not None and (
            not math.isfinite(effective_timeout) or effective_timeout <= 0
        ):
            raise ValueError("timeout_s must be finite and > 0 or None")
        deadline = None if effective_timeout is None else started + effective_timeout
        with self.cv:
            turn = self._enqueue_chat_locked(
                session_id=session_id,
                response_id=req_id,
                prompt_ids=prompt_ids,
                max_tokens=max_tokens,
                break_mask=break_mask,
                deadline=deadline,
                session_kind=_session_kind,
                input_mode=_input_mode,
                forced_output_ids=_forced_output_ids,
                chat_messages=chat_messages,
                chat_codec=chat_codec,
                parent_bound_contract=parent_bound_contract,
                assistant_message_id=assistant_message_id,
                user_message_id=user_message_id,
                parent_message_id=parent_message_id,
            )

        emitted = 0
        yield {"type": "queued", "req_id": req_id, "session_id": session_id}
        try:
            while True:
                timed_out = False
                with self.cv:
                    while True:
                        current_tokens = self._chat_turn_tokens(turn)
                        if len(current_tokens) != emitted or turn.finished:
                            break
                        wait_s = 0.25
                        if deadline is not None:
                            remaining = deadline - time.perf_counter()
                            if remaining <= 0:
                                timed_out = True
                                break
                            wait_s = min(wait_s, remaining)
                        self.cv.wait(timeout=wait_s)
                        if self.closed:
                            raise RuntimeError("service is closed")

                    if timed_out:
                        tokens = []
                        start_index = emitted
                        finished = False
                    else:
                        current_tokens = self._chat_turn_tokens(turn)
                        tokens = current_tokens[emitted:]
                        start_index = emitted
                        emitted = len(current_tokens)
                        finished = turn.finished

                if timed_out:
                    if self._cancel_chat_turn(turn, timed_out=True):
                        raise TimeoutError(f"request {req_id} timed out")
                    continue

                for offset, token in enumerate(tokens):
                    yield {
                        "type": "token",
                        "req_id": req_id,
                        "session_id": session_id,
                        "index": start_index + offset,
                        "token": int(token),
                    }
                if finished:
                    yield {
                        "type": "finish",
                        "req_id": req_id,
                        "session_id": session_id,
                        "finish_reason": turn.finish_reason,
                        "error": turn.error,
                        "latency_s": round(time.perf_counter() - started, 6),
                        "metrics": turn.metrics,
                    }
                    return
        finally:
            if not turn.finished:
                self._cancel_chat_turn(turn)

    def commit_chat_visible_output(
        self,
        *,
        session_id: str,
        response_id: str,
        text: str,
    ) -> None:
        with self.cv:
            with self.engine_lock:
                session = self._chat_sessions.get(session_id)
                if session is None:
                    raise ValueError(f"unknown chat session {session_id}")
                if session.parent_bound_contract is None:
                    raise ValueError(
                        f"chat session {session_id} has no parent-bound contract"
                    )
                if session.active_turn is not None:
                    raise SessionBusy(
                        f"chat session {session_id} has not completed its active turn"
                    )
                if session.last_response_id != response_id:
                    raise ValueError(
                        f"chat session {session_id} response identity changed"
                    )
                if session.canonical_input_messages is None:
                    raise ValueError(
                        f"chat session {session_id} has no completed input history"
                    )
                visible_output = str(text).strip()
                if session.visible_output is not None:
                    if session.visible_output != visible_output:
                        raise ValueError(
                            f"chat session {session_id} visible output changed"
                        )
                    return
                session.visible_output = visible_output
                session.visible_parent_history_digest = _message_history_digest(
                    [
                        *session.canonical_input_messages,
                        {"role": "assistant", "content": visible_output},
                    ]
                )
                session.generation += 1

    def stream_token_session(
        self,
        *,
        session_id: str,
        max_tokens: int,
        req_id: str,
        prompt_ids: list[int] | None = None,
        delta_ids: list[int] | None = None,
        forced_output_ids: list[int] | None = None,
        break_mask: list[bool] | None = None,
        timeout_s: float | None = None,
    ) -> Iterator[dict[str, Any]]:
        if (prompt_ids is None) == (delta_ids is None):
            raise ValueError(
                "stateful token request requires exactly one of prompt_ids or delta_ids"
            )
        input_mode = "initial_prompt" if prompt_ids is not None else "continuation_delta"
        input_ids = prompt_ids if prompt_ids is not None else delta_ids
        assert input_ids is not None
        if forced_output_ids is not None:
            if not self.enable_token_session_teacher_forcing:
                raise ValueError("token-session teacher forcing is disabled")
            forced_output_ids = [int(token_id) for token_id in forced_output_ids]
            if len(forced_output_ids) != int(max_tokens):
                raise ValueError(
                    "forced_output_ids length must equal max_tokens for this turn"
                )
        return self.stream_chat(
            prompt_ids=input_ids,
            max_tokens=max_tokens,
            session_id=session_id,
            req_id=req_id,
            break_mask=break_mask,
            timeout_s=timeout_s,
            _session_kind="token",
            _input_mode=input_mode,
            _forced_output_ids=forced_output_ids,
        )

    def status(self, req_id: str) -> dict[str, Any]:
        with self.cv:
            request = self._requests.get(req_id)
            if request is None:
                raise KeyError(f"unknown req_id {req_id}")
            with self.engine_lock:
                trace = self.engine.finished_traces.get(req_id)
            return {
                "req_id": req_id,
                "tokens": list(request.output_token_ids),
                "status": request.status.name.lower(),
                "finished": request.status.is_finished,
                "finish_reason": (
                    request.status.name.removeprefix("FINISHED_").lower()
                    if request.status.is_finished
                    else None
                ),
                "error": None if trace is None else trace.error,
                "metrics": trace.as_dict() if trace is not None else None,
            }

    def cancel(self, req_id: str) -> dict[str, Any]:
        cancelled = self._cancel_request(req_id)
        with self.cv:
            if cancelled:
                self.total_cancelled += 1
            self.cv.notify_all()
            return {"req_id": req_id, "cancelled": cancelled}

    def _timeout_request(self, req_id: str) -> bool:
        cancelled = self._cancel_request(req_id)
        with self.cv:
            if cancelled:
                self.total_timed_out += 1
            self.cv.notify_all()
        return cancelled

    def health(self) -> dict[str, Any]:
        with self.lock:
            worker_alive = self._worker.is_alive()
            ready = self.ready and not self.closed and worker_alive
            pending = len(self._pending) + len(self._pending_chat)
            with self.engine_lock:
                waiting = len(self.engine.scheduler.waiting)
                running = len(self.engine.scheduler.running)
                free_state_slots = self.engine.arena.num_free_slots()
                chat_sessions = len(self._chat_sessions)
                token_sessions = sum(
                    session.session_kind == "token"
                    for session in self._chat_sessions.values()
                )
            return {
                "ok": ready,
                "queue_depth": pending + waiting,
                "pending_queue_depth": pending,
                "running": running,
                "free_state_slots": free_state_slots,
                "chat_sessions": chat_sessions,
                "token_sessions": token_sessions,
                "timed_out_requests": self.total_timed_out,
                "worker_alive": worker_alive,
                "last_error": self.last_error,
            }

    def metrics(self) -> dict[str, Any]:
        with self.lock:
            worker_alive = self._worker.is_alive()
            ready = self.ready and not self.closed and worker_alive
            pending = len(self._pending) + len(self._pending_chat)
            with self.engine_lock:
                engine_stats = self.engine.stats()
                chat_sessions = len(self._chat_sessions)
                token_sessions = sum(
                    session.session_kind == "token"
                    for session in self._chat_sessions.values()
                )
            return {
                "server": {
                    "ready": ready,
                    "closed": self.closed,
                    "total_requests": self.total_requests,
                    "total_errors": self.total_errors,
                    "total_cancelled": self.total_cancelled,
                    "total_timed_out": self.total_timed_out,
                    "max_queue": self.max_queue,
                    "batch_wait_s": self.batch_wait_s,
                    "request_timeout_s": self.request_timeout_s,
                    "max_completed_requests": self.max_completed_requests,
                    "chat_session_ttl_s": self.chat_session_ttl_s,
                    "max_chat_sessions": self.max_chat_sessions,
                    "chat_sessions": chat_sessions,
                    "token_sessions": token_sessions,
                    "chat_exact_prefix_reuse_hits": (
                        self._chat_exact_prefix_reuse_hits
                    ),
                    "parent_bound_continuation_hits": (
                        self._parent_bound_continuation_hits
                    ),
                    "parent_bound_continuation_misses": (
                        self._parent_bound_continuation_misses
                    ),
                    "parent_bound_continuation_rejections": dict(
                        self._parent_bound_continuation_rejections
                    ),
                    "token_session_teacher_forcing_enabled": (
                        self.enable_token_session_teacher_forcing
                    ),
                    "empty_cuda_cache_before_decode": self.cuda_empty_cache is not None,
                    "cuda_empty_cache_calls": self.cuda_empty_cache_calls,
                    "pending_queue_depth": pending,
                    "tracked_requests": len(self._requests),
                    "completed_tracked_requests": len(self._completed_order),
                    "worker_alive": worker_alive,
                    "last_error": self.last_error,
                },
                "engine": engine_stats,
            }

    def close(self, *, timeout_s: float = 5.0) -> None:
        timeout_s = float(timeout_s)
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("timeout_s must be finite and > 0")
        with self.cv:
            self.closed = True
            self.ready = False
            self.cv.notify_all()
        if threading.current_thread() is self._worker:
            raise RuntimeError("engine worker cannot close its own service")
        self._worker.join(timeout=timeout_s)
        if self._worker.is_alive():
            raise RuntimeError(
                f"engine worker did not stop within {timeout_s:g} seconds"
            )
        with self.cv:
            self._fail_unfinished_locked("service is closed")
            self.cv.notify_all()

    def _run_engine(self) -> None:
        try:
            self._run_engine_loop()
        except BaseException as exc:
            error = f"worker terminated: {type(exc).__name__}: {exc}"
            with self.cv:
                self.ready = False
                self.total_errors += 1
                self.last_error = error
                try:
                    self._fail_unfinished_locked(error)
                except BaseException as cleanup_exc:
                    self.last_error = (
                        f"{error}; request cleanup failed: "
                        f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                    )
                    self._deadlines.clear()
                    for request in self._requests.values():
                        if request.status.is_finished:
                            continue
                        status_type = type(request.status)
                        request.status = getattr(
                            status_type,
                            "FINISHED_ERROR",
                            status_type.FINISHED_ABORTED,
                        )
                    self._record_completed_locked()
                    self._trim_completed_locked()
                finally:
                    self.cv.notify_all()

    def _run_engine_loop(self) -> None:
        while True:
            with self.cv:
                if (
                    not self.closed
                    and not self._pending
                    and not self._pending_chat
                    and not self.engine.has_unfinished
                ):
                    self.cv.wait(timeout=0.25)
                if self.closed:
                    return
                if (
                    self.batch_wait_s > 0
                    and not self.engine.has_unfinished
                    and (self._pending or self._pending_chat)
                ):
                    batch_deadline = time.monotonic() + self.batch_wait_s
                    while not self.closed:
                        remaining = batch_deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self.cv.wait(timeout=remaining)
                    if self.closed:
                        return
                self._expire_deadlines_locked()
                pending = list(self._pending)
                self._pending.clear()
                pending_chat = list(self._pending_chat)
                self._pending_chat.clear()
            finalized_chat: list[_ChatTurn] = []
            failed_chat: list[_ChatTurn] = []
            retry_chat: list[_ChatTurn] = []
            try:
                with self.engine_lock:
                    self._evict_expired_chat_sessions_locked(time.monotonic())
                    for request, break_mask in pending:
                        if request.req_id in self.cancelled:
                            self.cancelled.discard(request.req_id)
                            request.status = type(request.status).FINISHED_ABORTED
                            continue
                        self.engine.add_request(request, break_mask=break_mask)
                    for turn in pending_chat:
                        if turn.cancel_requested or turn.finished:
                            continue
                        try:
                            if not self._start_chat_turn_locked(turn):
                                retry_chat.append(turn)
                        except Exception as exc:
                            self._fail_chat_turn_engine_locked(turn, str(exc))
                            failed_chat.append(turn)
                            self.total_errors += 1
                    if self.engine.has_unfinished:
                        completed = self.engine.step() or []
                        self._force_token_session_outputs_locked()
                        self._maybe_empty_cuda_cache_before_decode_locked()
                        finalized_chat.extend(
                            self._completed_chat_turns_locked(completed)
                        )
            except Exception as exc:
                with self.cv:
                    self.total_errors += 1
                    self.last_error = str(exc)
                    self._fail_unfinished_locked(str(exc).splitlines()[0])
                    self.cv.notify_all()
            else:
                with self.cv:
                    for turn in failed_chat:
                        self._finish_chat_turn_locked(turn)
                    for turn in finalized_chat:
                        self._finish_chat_turn_locked(turn)
                    for turn in retry_chat:
                        if not turn.cancel_requested and not turn.finished:
                            self._pending_chat.append(turn)
                    self._expire_deadlines_locked()
                    self._record_completed_locked()
                    self._trim_completed_locked()
                    self.cv.notify_all()

    def _force_token_session_outputs_locked(self) -> None:
        for session in self._chat_sessions.values():
            turn = session.active_turn
            if turn is None or turn.forced_output_token_ids is None:
                continue
            output_tokens = session.request.output_token_ids
            while len(turn.candidate_output_token_ids) < len(output_tokens):
                output_index = len(turn.candidate_output_token_ids)
                if output_index >= len(turn.forced_output_token_ids):
                    raise RuntimeError(
                        f"token session {turn.session_id} produced more tokens "
                        "than forced_output_ids"
                    )
                turn.candidate_output_token_ids.append(
                    int(output_tokens[output_index])
                )
                overwrite_started = time.perf_counter()
                output_tokens[output_index] = int(
                    turn.forced_output_token_ids[output_index]
                )
                turn.teacher_forcing_overwrite_s += (
                    time.perf_counter() - overwrite_started
                )
                turn.teacher_forcing_overwrites += 1

    @staticmethod
    def _stateful_turn_metrics(
        turn: _ChatTurn,
        engine_metrics: dict[str, Any] | None,
    ) -> dict[str, Any]:
        metrics = {} if engine_metrics is None else dict(engine_metrics)
        metrics.update(
            {
                "http_session_id": turn.session_id,
                "session_kind": turn.session_kind,
                "session_input_mode": turn.input_mode,
                "session_input_tokens": len(turn.prompt_token_ids),
                "session_reused": turn.session_reused,
                "session_reuse_mode": turn.session_reuse_mode,
                "parent_bound_continuation": {
                    "enabled": turn.parent_bound_contract is not None,
                    "contract": turn.parent_bound_contract,
                    "attempted": turn.parent_bound_continuation_attempted,
                    "accepted": turn.parent_bound_continuation_accepted,
                    "rejection_reason": (
                        turn.parent_bound_continuation_rejection_reason
                    ),
                },
            }
        )
        forced = turn.forced_output_token_ids
        if forced is None:
            metrics["teacher_forcing"] = {"enabled": False}
        else:
            selected = list(turn.output_token_ids)
            candidates = list(turn.candidate_output_token_ids)
            metrics["teacher_forcing"] = {
                "enabled": True,
                "backend": "post_sample_pending_token_override",
                "overhead_contract": {
                    "timed": True,
                    "full_vocabulary_mask": False,
                    "gpu_logit_elements_mutated_per_row": 0,
                    "mutation": "one_pending_token_scalar_overwrite",
                    "row_mutation_scope": "request_loop",
                },
                "forced_output_ids": list(forced),
                "candidate_output_ids": candidates,
                "selected_outputs_match_forced": selected == forced,
                "candidate_outputs_match_forced": candidates == forced,
                "scalar_overwrite_count": turn.teacher_forcing_overwrites,
                "scalar_overwrite_s": round(turn.teacher_forcing_overwrite_s, 9),
            }
        return metrics

    def _maybe_empty_cuda_cache_before_decode_locked(self) -> None:
        if self.cuda_empty_cache is None:
            return
        active_turns = [
            session.active_turn
            for session in self._chat_sessions.values()
            if session.active_turn is not None
            and not session.active_turn.cuda_cache_emptied
        ]
        if not active_turns or any(
            turn.request is None or not turn.request.output_token_ids
            for turn in active_turns
        ):
            return
        self.cuda_empty_cache()
        for turn in active_turns:
            turn.cuda_cache_emptied = True
        self.cuda_empty_cache_calls += 1

    @staticmethod
    def _chat_turn_tokens(turn: _ChatTurn) -> list[int]:
        if turn.forced_output_token_ids is not None:
            safe_count = len(turn.candidate_output_token_ids)
            return list(turn.forced_output_token_ids[:safe_count])
        if turn.finished:
            return list(turn.output_token_ids)
        if turn.request is None or turn.state != "running":
            return []
        return list(turn.request.output_token_ids)

    def _enqueue_chat_locked(
        self,
        *,
        session_id: str,
        response_id: str,
        prompt_ids: list[int],
        max_tokens: int,
        break_mask: list[bool] | None,
        deadline: float | None,
        session_kind: str = "chat",
        input_mode: str = "full_prompt",
        forced_output_ids: list[int] | None = None,
        chat_messages: list[dict[str, str]] | None = None,
        chat_codec: Any | None = None,
        parent_bound_contract: str | None = None,
        assistant_message_id: str | None = None,
        user_message_id: str | None = None,
        parent_message_id: str | None = None,
    ) -> _ChatTurn:
        if self.closed:
            self.total_errors += 1
            raise ServiceUnavailable("service is closed")
        if not self.ready or not self._worker.is_alive():
            self.total_errors += 1
            detail = self.last_error or "engine worker is not running"
            raise ServiceUnavailable(f"service is not ready: {detail}")
        session_id = str(session_id).strip()
        if not session_id:
            raise ValueError("session_id must not be empty")
        if session_kind not in {"chat", "token"}:
            raise ValueError(f"unsupported session kind {session_kind!r}")
        allowed_input_modes = (
            {"full_prompt"}
            if session_kind == "chat"
            else {"initial_prompt", "continuation_delta"}
        )
        if input_mode not in allowed_input_modes:
            raise ValueError(
                f"unsupported {session_kind} session input mode {input_mode!r}"
            )
        assistant_message_id = _chat_identity(
            assistant_message_id,
            "assistant_message_id",
        )
        user_message_id = _chat_identity(user_message_id, "user_message_id")
        parent_message_id = _chat_identity(
            parent_message_id,
            "parent_message_id",
        )
        if parent_bound_contract is not None:
            parent_bound_contract = str(parent_bound_contract).strip()
            if parent_bound_contract != PARENT_TOKEN_CHAT_CONTRACT:
                raise ValueError(
                    f"unsupported parent-bound chat contract {parent_bound_contract!r}"
                )
            if session_kind != "chat":
                raise ValueError(
                    "parent-bound chat metadata is only valid for chat sessions"
                )
            if chat_messages is None or chat_codec is None:
                raise ValueError(
                    "parent-bound chat requires normalized messages and a codec"
                )
            if assistant_message_id is None or user_message_id is None:
                raise ValueError(
                    "parent-bound chat requires assistant and user message IDs"
                )
        elif any(
            value is not None
            for value in (
                assistant_message_id,
                user_message_id,
                parent_message_id,
            )
        ):
            raise ValueError(
                "chat message identity headers require a parent-bound contract"
            )
        if forced_output_ids is not None:
            if session_kind != "token":
                raise ValueError("teacher forcing is only available for token sessions")
            if not self.enable_token_session_teacher_forcing:
                raise ValueError("token-session teacher forcing is disabled")
            forced_output_ids = [int(token_id) for token_id in forced_output_ids]
            if len(forced_output_ids) != int(max_tokens):
                raise ValueError(
                    "forced_output_ids length must equal max_tokens for this turn"
                )
        if session_id in self._chat_active_turns:
            raise SessionBusy(f"session {session_id} already has an active turn")
        prompt_ids = [int(token_id) for token_id in prompt_ids]
        if not prompt_ids:
            raise ValueError("session input token ids must not be empty")
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        with self.engine_lock:
            session = self._chat_sessions.get(session_id)
            if session is not None and session.session_kind != session_kind:
                raise ValueError(
                    f"session {session_id} belongs to the {session.session_kind} API"
                )
            if session_kind == "token":
                if input_mode == "initial_prompt" and session is not None:
                    raise ValueError(
                        f"token session {session_id} already exists; send delta_ids"
                    )
                if input_mode == "continuation_delta":
                    if session is None:
                        raise ValueError(
                            f"unknown token session {session_id}; send prompt_ids first"
                        )
                    parked = getattr(self.engine.scheduler, "parked", {})
                    if (
                        session.request.status.name != "PARKED"
                        or parked.get(session.engine_req_id) is not session.request
                    ):
                        raise SessionBusy(
                            f"token session {session_id} is not parked for continuation"
                        )
                    session.last_access = time.monotonic()
            queued = (
                len(self._pending)
                + len(self._pending_chat)
                + len(self.engine.scheduler.waiting)
            )
        if queued >= self.max_queue:
            self.total_errors += 1
            raise QueueFull("bounded request queue is full")
        turn = _ChatTurn(
            session_id=session_id,
            response_id=response_id,
            prompt_token_ids=prompt_ids,
            max_new_tokens=int(max_tokens),
            break_mask=None if break_mask is None else list(break_mask),
            deadline=deadline,
            session_kind=session_kind,
            input_mode=input_mode,
            forced_output_token_ids=forced_output_ids,
            chat_messages=(
                None
                if chat_messages is None
                else [dict(message) for message in chat_messages]
            ),
            chat_codec=chat_codec,
            parent_bound_contract=parent_bound_contract,
            assistant_message_id=assistant_message_id,
            user_message_id=user_message_id,
            parent_message_id=parent_message_id,
        )
        self.total_requests += 1
        self._pending_chat.append(turn)
        self._chat_active_turns[session_id] = turn
        self.cv.notify_all()
        return turn

    def _resume_chat_session_locked(
        self,
        session: _ChatSession,
        turn: _ChatTurn,
        continuation: list[int],
        *,
        break_mask: list[bool] | None = None,
        reuse_mode: str,
    ) -> bool:
        scheduler = self.engine.scheduler
        max_running = getattr(
            getattr(scheduler, "config", None),
            "max_running_requests",
            None,
        )
        if max_running is not None and len(scheduler.running) >= max_running:
            return False
        self.engine.continue_session_requests(
            {session.engine_req_id: continuation},
            max_new_tokens=turn.max_new_tokens,
            break_masks={
                session.engine_req_id: (
                    turn.break_mask if break_mask is None else break_mask
                )
            },
        )
        session.active_turn = turn
        session.last_access = time.monotonic()
        turn.request = session.request
        turn.engine_req_id = session.engine_req_id
        turn.state = "running"
        turn.session_reused = True
        turn.session_reuse_mode = reuse_mode
        return True

    def _record_parent_bound_rejection_locked(
        self,
        turn: _ChatTurn,
        reason: str,
    ) -> None:
        turn.parent_bound_continuation_attempted = True
        turn.parent_bound_continuation_accepted = False
        turn.parent_bound_continuation_rejection_reason = reason
        turn.session_reuse_mode = "restart"
        self._parent_bound_continuation_misses += 1
        self._parent_bound_continuation_rejections[reason] = (
            self._parent_bound_continuation_rejections.get(reason, 0) + 1
        )

    @staticmethod
    def _validate_parent_bound_history(
        session: _ChatSession,
        turn: _ChatTurn,
        retained: list[int],
    ) -> list[dict[str, str]] | str:
        turn.parent_bound_continuation_attempted = True
        if session.parent_bound_contract != turn.parent_bound_contract:
            return "contract_mismatch"
        if (
            turn.chat_messages is None
            or turn.chat_codec is None
            or turn.assistant_message_id is None
            or turn.user_message_id is None
        ):
            return "request_metadata_unavailable"
        if (
            session.canonical_input_messages is None
            or session.canonical_prompt_token_ids is None
            or session.visible_output is None
            or session.visible_parent_history_digest is None
            or session.retained_token_digest is None
            or session.last_assistant_message_id is None
            or session.last_user_message_id is None
        ):
            return "session_history_unavailable"
        if turn.parent_message_id != session.last_assistant_message_id:
            return "parent_message_mismatch"
        if turn.assistant_message_id == session.last_assistant_message_id:
            return "assistant_message_reused"
        if turn.user_message_id == session.last_user_message_id:
            return "user_message_reused"
        if _token_history_digest(retained) != session.retained_token_digest:
            return "retained_history_mismatch"

        previous_messages = session.canonical_input_messages
        incoming_messages = turn.chat_messages
        previous_count = len(previous_messages)
        if incoming_messages[:previous_count] != previous_messages:
            return "prior_history_mismatch"
        if len(incoming_messages) <= previous_count:
            return "prior_assistant_missing"
        expected_assistant = {
            "role": "assistant",
            "content": session.visible_output,
        }
        if incoming_messages[previous_count] != expected_assistant:
            return "prior_assistant_mismatch"
        parent_history = incoming_messages[: previous_count + 1]
        if (
            _message_history_digest(parent_history)
            != session.visible_parent_history_digest
        ):
            return "parent_history_digest_mismatch"
        appended_messages = incoming_messages[previous_count + 1 :]
        if not appended_messages:
            return "new_messages_missing"
        return appended_messages

    @staticmethod
    def _parent_bound_continuation(
        session: _ChatSession,
        turn: _ChatTurn,
        retained: list[int],
        appended_messages: list[dict[str, str]],
    ) -> tuple[list[int], list[bool]] | str:
        assert turn.chat_codec is not None
        assert session.canonical_input_messages is not None
        assert session.canonical_prompt_token_ids is not None
        bridge_messages = [
            *session.canonical_input_messages,
            {"role": "assistant", "content": ""},
            *appended_messages,
        ]
        try:
            extended = turn.chat_codec.prompt_token_ids(bridge_messages)
        except Exception:
            return "template_render_failed"
        base = session.canonical_prompt_token_ids
        if len(extended) <= len(base) or extended[: len(base)] != base:
            return "template_prefix_mismatch"
        continuation = extended[len(base) :]
        generated = session.request.output_token_ids
        if generated and continuation and generated[-1] == continuation[0]:
            try:
                terminal_boundary_is_hidden = not turn.chat_codec.decode(
                    [continuation[0]]
                )
            except Exception:
                return "boundary_decode_failed"
            if terminal_boundary_is_hidden:
                continuation = continuation[1:]
        if not continuation:
            return "continuation_empty"
        try:
            break_mask = turn.chat_codec.break_mask([*retained, *continuation])
        except Exception:
            return "break_mask_failed"
        return continuation, break_mask

    def _start_chat_turn_locked(self, turn: _ChatTurn) -> bool:
        from wkvm.core.request import Request

        session = self._chat_sessions.get(turn.session_id)
        if session is not None and session.session_kind != turn.session_kind:
            raise ValueError(
                f"session {turn.session_id} belongs to the {session.session_kind} API"
            )
        if session is not None and session.active_turn is not None:
            return False
        if turn.session_kind == "token" and turn.input_mode == "continuation_delta":
            if session is None:
                raise ValueError(f"unknown token session {turn.session_id}")
            if session.request.status.name != "PARKED":
                raise SessionBusy(
                    f"token session {turn.session_id} is not parked for continuation"
                )
            return self._resume_chat_session_locked(
                session,
                turn,
                list(turn.prompt_token_ids),
                reuse_mode="token_delta",
            )
        if turn.session_kind == "token" and session is not None:
            raise ValueError(
                f"token session {turn.session_id} already exists; send delta_ids"
            )
        parent_rejection_recorded = False
        if session is not None and session.request.status.name != "PARKED":
            if turn.parent_bound_contract is not None:
                self._record_parent_bound_rejection_locked(
                    turn,
                    "session_not_parked",
                )
                parent_rejection_recorded = True
            self._drop_chat_session_locked(session)
            session = None
        if session is not None:
            retained = (
                list(session.request.prompt_token_ids)
                + list(session.request.output_token_ids)
            )
            parent_bound = (
                session.parent_bound_contract is not None
                or turn.parent_bound_contract is not None
            )
            appended_messages: list[dict[str, str]] | None = None
            if parent_bound:
                validation = self._validate_parent_bound_history(
                    session,
                    turn,
                    retained,
                )
                if isinstance(validation, str):
                    self._record_parent_bound_rejection_locked(turn, validation)
                    parent_rejection_recorded = True
                    self._retire_chat_session_locked(session)
                    session = None
                else:
                    appended_messages = validation

            if session is not None:
                exact_prefix = (
                    len(turn.prompt_token_ids) > len(retained)
                    and turn.prompt_token_ids[: len(retained)] == retained
                )
                if exact_prefix:
                    continuation = turn.prompt_token_ids[len(retained) :]
                    turn.parent_bound_continuation_accepted = parent_bound
                    resumed = self._resume_chat_session_locked(
                        session,
                        turn,
                        continuation,
                        reuse_mode="exact_prefix",
                    )
                    if resumed:
                        self._chat_exact_prefix_reuse_hits += 1
                    return resumed
                if parent_bound:
                    assert appended_messages is not None
                    parent_continuation = self._parent_bound_continuation(
                        session,
                        turn,
                        retained,
                        appended_messages,
                    )
                    if isinstance(parent_continuation, str):
                        self._record_parent_bound_rejection_locked(
                            turn,
                            parent_continuation,
                        )
                        parent_rejection_recorded = True
                        self._retire_chat_session_locked(session)
                        session = None
                    else:
                        continuation, parent_break_mask = parent_continuation
                        turn.parent_bound_continuation_accepted = True
                        resumed = self._resume_chat_session_locked(
                            session,
                            turn,
                            continuation,
                            break_mask=parent_break_mask,
                            reuse_mode="parent_bound_delta",
                        )
                        if resumed:
                            self._parent_bound_continuation_hits += 1
                        return resumed
                else:
                    self._retire_chat_session_locked(session)
                    session = None

        if (
            session is None
            and turn.parent_bound_contract is not None
            and turn.parent_message_id is not None
            and not parent_rejection_recorded
        ):
            self._record_parent_bound_rejection_locked(
                turn,
                "session_unavailable",
            )

        if not self._make_chat_session_room_locked():
            return False
        self._chat_req_counter += 1
        engine_req_id = f"wkvm-chat-session-{self._chat_req_counter}"
        request = Request(
            prompt_token_ids=list(turn.prompt_token_ids),
            max_new_tokens=turn.max_new_tokens,
            req_id=engine_req_id,
        )
        self.engine.add_session_request(request, break_mask=turn.break_mask)
        session = _ChatSession(
            session_id=turn.session_id,
            engine_req_id=engine_req_id,
            session_kind=turn.session_kind,
            request=request,
            active_turn=turn,
            last_access=time.monotonic(),
            parent_bound_contract=turn.parent_bound_contract,
        )
        self._chat_sessions[turn.session_id] = session
        self._chat_sessions_by_engine_id[engine_req_id] = session
        turn.request = request
        turn.engine_req_id = engine_req_id
        turn.state = "running"
        return True

    def _completed_chat_turns_locked(self, completed) -> list[_ChatTurn]:
        finalized: list[_ChatTurn] = []
        for request in completed:
            session = self._chat_sessions_by_engine_id.get(request.req_id)
            if session is None or session.active_turn is None:
                continue
            turn = session.active_turn
            turn.output_token_ids = list(request.output_token_ids)
            trace = self.engine.finished_traces.get(request.req_id)
            engine_metrics = None
            if trace is not None:
                turn.error = trace.error
                turn.finish_reason = trace.finish_reason
                engine_metrics = trace.as_dict()
            else:
                terminal_status = request.parked_finish_status or request.status
                turn.finish_reason = terminal_status.name.removeprefix("FINISHED_").lower()
            forced = turn.forced_output_token_ids
            forcing_mismatch = forced is not None and turn.output_token_ids != forced
            if forcing_mismatch:
                turn.error = "selected outputs did not match forced_output_ids"
                turn.finish_reason = "error"
            session.active_turn = None
            session.last_access = time.monotonic()
            if (
                turn.parent_bound_contract is not None
                and request.status.name == "PARKED"
                and not forcing_mismatch
            ):
                assert turn.chat_messages is not None
                assert turn.assistant_message_id is not None
                assert turn.user_message_id is not None
                session.parent_bound_contract = turn.parent_bound_contract
                session.canonical_input_messages = [
                    dict(message) for message in turn.chat_messages
                ]
                session.canonical_prompt_token_ids = list(turn.prompt_token_ids)
                session.visible_output = None
                session.visible_parent_history_digest = None
                session.retained_token_digest = _token_history_digest(
                    [
                        *session.request.prompt_token_ids,
                        *session.request.output_token_ids,
                    ]
                )
                session.last_assistant_message_id = turn.assistant_message_id
                session.last_user_message_id = turn.user_message_id
                session.last_response_id = turn.response_id
            turn.metrics = self._stateful_turn_metrics(turn, engine_metrics)
            if forcing_mismatch:
                self._retire_chat_session_locked(session)
            elif request.status.name != "PARKED":
                self._drop_chat_session_locked(session)
            finalized.append(turn)
        return finalized

    def _finish_chat_turn_locked(self, turn: _ChatTurn) -> None:
        turn.state = "finished"
        current = self._chat_active_turns.get(turn.session_id)
        if current is turn:
            self._chat_active_turns.pop(turn.session_id, None)

    def _fail_chat_turn_engine_locked(self, turn: _ChatTurn, error: str) -> None:
        session = self._chat_sessions.get(turn.session_id)
        if session is not None and session.active_turn is turn:
            if not session.request.status.is_finished:
                self.engine.abort_request(session.engine_req_id)
            self._drop_chat_session_locked(session)
        turn.error = error
        turn.finish_reason = "error"
        turn.metrics = self._stateful_turn_metrics(turn, turn.metrics)

    def _make_chat_session_room_locked(self) -> bool:
        while True:
            over_limit = (
                self.max_chat_sessions is not None
                and len(self._chat_sessions) >= self.max_chat_sessions
            )
            no_slot = self.engine.arena.num_free_slots() < 1
            if not over_limit and not no_slot:
                return True
            candidates = [
                session
                for session in self._chat_sessions.values()
                if session.active_turn is None and session.request.status.name == "PARKED"
                and session.session_id not in self._chat_active_turns
            ]
            if not candidates:
                return False
            self._retire_chat_session_locked(
                min(candidates, key=lambda session: session.last_access)
            )

    def _evict_expired_chat_sessions_locked(self, now: float) -> None:
        if self.chat_session_ttl_s is None:
            return
        expired = [
            session
            for session in self._chat_sessions.values()
            if session.active_turn is None
            and session.request.status.name == "PARKED"
            and session.session_id not in self._chat_active_turns
            and now - session.last_access >= self.chat_session_ttl_s
        ]
        for session in expired:
            self._retire_chat_session_locked(session)

    def _retire_chat_session_locked(self, session: _ChatSession) -> None:
        if session.request.status.name == "PARKED":
            self.engine.close_sessions([session.engine_req_id])
        elif not session.request.status.is_finished:
            self.engine.abort_request(session.engine_req_id)
        self._drop_chat_session_locked(session)

    def _drop_chat_session_locked(self, session: _ChatSession) -> None:
        if self._chat_sessions.get(session.session_id) is session:
            self._chat_sessions.pop(session.session_id, None)
        self._chat_sessions_by_engine_id.pop(session.engine_req_id, None)
        requests = getattr(self.engine.scheduler, "requests", None)
        if requests is not None:
            requests.pop(session.engine_req_id, None)

    def _cancel_chat_turn(self, turn: _ChatTurn, *, timed_out: bool = False) -> bool:
        with self.cv:
            if turn.finished:
                return False
            turn.cancel_requested = True
            try:
                self._pending_chat.remove(turn)
            except ValueError:
                pass
            engine_req_id = turn.engine_req_id

        completed_race = False
        if engine_req_id is not None:
            with self.engine_lock:
                session = self._chat_sessions_by_engine_id.get(engine_req_id)
                if session is not None:
                    if session.request.status.name == "PARKED":
                        completed_race = True
                        turn.output_token_ids = list(session.request.output_token_ids)
                        trace = self.engine.finished_traces.get(engine_req_id)
                        if trace is not None:
                            turn.error = trace.error
                            turn.finish_reason = trace.finish_reason
                            turn.metrics = trace.as_dict()
                        else:
                            terminal = session.request.parked_finish_status
                            turn.finish_reason = (
                                None
                                if terminal is None
                                else terminal.name.removeprefix("FINISHED_").lower()
                            )
                        session.active_turn = None
                        session.last_access = time.monotonic()
                    else:
                        self.engine.abort_request(engine_req_id)
                        self._drop_chat_session_locked(session)
                        turn.finish_reason = "aborted"
        turn.metrics = self._stateful_turn_metrics(turn, turn.metrics)
        if turn.finish_reason is None:
            turn.finish_reason = "aborted"
        with self.cv:
            if not turn.finished:
                self._finish_chat_turn_locked(turn)
                if completed_race:
                    pass
                elif timed_out:
                    self.total_timed_out += 1
                else:
                    self.total_cancelled += 1
                self.cv.notify_all()
                return not completed_race
        return False

    def _enqueue_locked(self, request, *, break_mask: list[bool] | None):
        if self.closed:
            self.total_errors += 1
            raise ServiceUnavailable("service is closed")
        if not self.ready or not self._worker.is_alive():
            self.total_errors += 1
            detail = self.last_error or "engine worker is not running"
            raise ServiceUnavailable(f"service is not ready: {detail}")
        if request.req_id in self._requests:
            self.total_errors += 1
            raise ValueError(f"duplicate req_id {request.req_id}")
        with self.engine_lock:
            queued = (
                len(self._pending)
                + len(self._pending_chat)
                + len(self.engine.scheduler.waiting)
            )
        if queued >= self.max_queue:
            self.total_errors += 1
            raise QueueFull("bounded request queue is full")
        self.total_requests += 1
        self._pending.append((request, break_mask))
        self._requests[request.req_id] = request
        self.cv.notify_all()
        return request

    def _cancel_locked(self, req_id: str, *, timed_out: bool = False) -> bool:
        request = self._requests.get(req_id)
        if request is None:
            raise KeyError(f"unknown req_id {req_id}")
        if request.status.is_finished:
            self.cancelled.discard(req_id)
            self._deadlines.pop(req_id, None)
            return False
        self.cancelled.add(req_id)
        self._deadlines.pop(req_id, None)
        for item in list(self._pending):
            pending_request, _ = item
            if pending_request.req_id == req_id:
                self._pending.remove(item)
                pending_request.status = type(pending_request.status).FINISHED_ABORTED
                self.cancelled.discard(req_id)
                self._record_completed_locked(req_id)
                self._trim_completed_locked()
                if timed_out:
                    self.total_timed_out += 1
                return True
        with self.engine_lock:
            if request.status.is_finished:
                self.cancelled.discard(req_id)
                self._record_completed_locked(req_id)
                self._trim_completed_locked()
                return False
            self.engine.abort_request(req_id)
        if request.status.is_finished:
            self.cancelled.discard(req_id)
            self._record_completed_locked(req_id)
            self._trim_completed_locked()
        if timed_out:
            self.total_timed_out += 1
        return True

    def _cancel_request(self, req_id: str) -> bool:
        with self.cv:
            request = self._requests.get(req_id)
            if request is None:
                raise KeyError(f"unknown req_id {req_id}")
            if request.status.is_finished:
                self.cancelled.discard(req_id)
                self._deadlines.pop(req_id, None)
                return False
            self._deadlines.pop(req_id, None)
            for item in list(self._pending):
                pending_request, _ = item
                if pending_request.req_id != req_id:
                    continue
                self._pending.remove(item)
                pending_request.status = type(
                    pending_request.status
                ).FINISHED_ABORTED
                self.cancelled.discard(req_id)
                self._record_completed_locked(req_id)
                self._trim_completed_locked()
                return True
            self.cancelled.add(req_id)

        with self.engine_lock:
            if request.status.is_finished:
                cancelled = request.status.name == "FINISHED_ABORTED"
            else:
                self.engine.abort_request(req_id)
                cancelled = (
                    not request.status.is_finished
                    or request.status.name == "FINISHED_ABORTED"
                )

        with self.cv:
            if request.status.is_finished:
                self.cancelled.discard(req_id)
                self._record_completed_locked(req_id)
                self._trim_completed_locked()
            self.cv.notify_all()
        return cancelled

    def _fail_unfinished_locked(self, error: str) -> None:
        self._pending.clear()
        self._pending_chat.clear()
        with self.engine_lock:
            fail_unfinished = getattr(self.engine, "fail_unfinished", None)
            if callable(fail_unfinished):
                fail_unfinished(error)
            for session in list(self._chat_sessions.values()):
                self._drop_chat_session_locked(session)
        self._deadlines.clear()
        for request in self._requests.values():
            if request.status.is_finished:
                continue
            status_type = type(request.status)
            request.status = getattr(
                status_type,
                "FINISHED_ERROR",
                status_type.FINISHED_ABORTED,
            )
        for turn in list(self._chat_active_turns.values()):
            turn.error = error
            turn.finish_reason = "error"
            self._finish_chat_turn_locked(turn)
        self._record_completed_locked()
        self._trim_completed_locked()

    def _record_completed_locked(self, req_id: str | None = None) -> None:
        ids = [req_id] if req_id is not None else list(self._requests)
        known = set(self._completed_order)
        for rid in ids:
            request = self._requests.get(rid)
            if request is not None and request.status.is_finished and rid not in known:
                self._completed_order.append(rid)
                self.cancelled.discard(rid)
                self._deadlines.pop(rid, None)
                known.add(rid)

    def _trim_completed_locked(self) -> None:
        if self.max_completed_requests is None:
            return
        while len(self._completed_order) > self.max_completed_requests:
            req_id = self._completed_order.popleft()
            request = self._requests.get(req_id)
            if request is not None and request.status.is_finished:
                self._requests.pop(req_id, None)

    def _expire_deadlines_locked(self) -> None:
        if not self._deadlines:
            return
        now = time.perf_counter()
        expired = [req_id for req_id, deadline in self._deadlines.items() if deadline <= now]
        for req_id in expired:
            self._cancel_locked(req_id, timed_out=True)


class QueueFull(RuntimeError):
    pass


class ServiceUnavailable(RuntimeError):
    pass


class SessionBusy(RuntimeError):
    pass


def _openai_finish_reason(reason: str | None) -> str | None:
    if reason == "stopped":
        return "stop"
    if reason in {"length", "error", "aborted"}:
        return reason
    return reason


def _bool_field(body: dict[str, Any], name: str, default: bool) -> bool:
    value = body.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _token_id_list_field(body: dict[str, Any], name: str) -> list[int]:
    value = body.get(name)
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty token-id list")
    if not all(type(token_id) is int and token_id >= 0 for token_id in value):
        raise ValueError(f"{name} must be a token-id list")
    return list(value)


def _openai_completion_request(
    body: dict[str, Any],
    req_id: str | None,
    *,
    server_ignore_eos: bool = True,
) -> dict[str, Any]:
    if "prompt" not in body:
        raise KeyError("prompt")
    prompt = body["prompt"]
    if not isinstance(prompt, list) or not prompt:
        raise ValueError("prompt must be a non-empty token-id list")
    if isinstance(prompt[0], list):
        raise ValueError("batched prompts are not supported")
    if not all(type(tok) is int and tok >= 0 for tok in prompt):
        raise ValueError("prompt must be a token-id list")

    max_tokens = int(body.get("max_tokens", 16))
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    if int(body.get("n", 1)) != 1:
        raise ValueError("only n=1 is supported")
    if body.get("best_of") not in (None, 1):
        raise ValueError("best_of is not supported")
    if body.get("echo") not in (None, False):
        raise ValueError("echo is not supported")
    if body.get("logprobs") is not None:
        raise ValueError("logprobs is not supported")
    if body.get("suffix") is not None:
        raise ValueError("suffix is not supported")
    if body.get("stop") not in (None, [], ""):
        raise ValueError("per-request stop sequences are not supported")

    temperature = body.get("temperature", 0.0)
    if temperature is not None and float(temperature) != 0.0:
        raise ValueError("only greedy temperature=0 is supported")
    top_p = body.get("top_p", 1.0)
    if top_p is not None and float(top_p) != 1.0:
        raise ValueError("only top_p=1 is supported")
    ignore_eos = _bool_field(body, "ignore_eos", server_ignore_eos)
    if ignore_eos != server_ignore_eos:
        raise ValueError(
            "per-request ignore_eos must match the server EOS policy; "
            "restart with or without --ignore-eos"
        )

    stream_options = body.get("stream_options") or {}
    if not isinstance(stream_options, dict):
        raise ValueError("stream_options must be an object")
    include_usage = stream_options.get("include_usage", False)
    if not isinstance(include_usage, bool):
        raise ValueError("stream_options.include_usage must be a boolean")

    timeout_s = body.get("timeout_s")
    if timeout_s is not None:
        timeout_s = float(timeout_s)

    return {
        "model": str(body.get("model") or "wkvm-gemma"),
        "prompt_ids": list(prompt),
        "max_tokens": max_tokens,
        "stream": _bool_field(body, "stream", False),
        "include_usage": include_usage,
        "return_token_ids": _bool_field(body, "return_token_ids", False),
        "req_id": str(body.get("request_id") or body.get("req_id") or req_id or f"cmpl-{time.time_ns()}"),
        "timeout_s": timeout_s,
    }


def _openai_usage(prompt_tokens: int, completion_tokens: int) -> dict[str, int]:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def _openai_completion_choice(
    *,
    index: int = 0,
    finish_reason: str | None = None,
    token_ids: list[int] | None = None,
    return_token_ids: bool = False,
) -> dict[str, Any]:
    choice: dict[str, Any] = {
        "index": index,
        "text": "",
        "logprobs": None,
        "finish_reason": _openai_finish_reason(finish_reason),
    }
    if return_token_ids:
        choice["token_ids"] = [] if token_ids is None else list(token_ids)
    return choice


def _openai_completion_response(
    *,
    req_id: str,
    model: str,
    prompt_tokens: int,
    output_tokens: list[int],
    finish_reason: str | None,
    return_token_ids: bool,
) -> dict[str, Any]:
    return {
        "id": req_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            _openai_completion_choice(
                finish_reason=finish_reason,
                token_ids=output_tokens,
                return_token_ids=return_token_ids,
            )
        ],
        "usage": _openai_usage(prompt_tokens, len(output_tokens)),
    }


class _OpenAIChatCodec:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        self._break_cache: dict[int, bool] = {}
        self._break_lock = threading.Lock()

    def prompt_token_ids(self, messages: list[dict[str, Any]]) -> list[int]:
        kwargs = {
            "add_generation_prompt": True,
            "tokenize": True,
            "return_dict": False,
        }
        try:
            encoded = self.tokenizer.apply_chat_template(messages, **kwargs)
        except TypeError:
            kwargs.pop("return_dict")
            encoded = self.tokenizer.apply_chat_template(messages, **kwargs)
        if isinstance(encoded, dict):
            encoded = encoded.get("input_ids")
        elif hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        if (
            not isinstance(encoded, (list, tuple))
            or not encoded
            or isinstance(encoded[0], (list, tuple))
        ):
            raise ValueError("chat template must produce one non-empty token-id list")
        token_ids = [int(token_id) for token_id in encoded]
        if any(token_id < 0 for token_id in token_ids):
            raise ValueError("chat template produced an invalid token id")
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        try:
            return str(
                self.tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
        except TypeError:
            return str(self.tokenizer.decode(token_ids, skip_special_tokens=True))

    def break_mask(self, token_ids: list[int]) -> list[bool]:
        missing = set(token_ids).difference(self._break_cache)
        if missing:
            with self._break_lock:
                for token_id in missing.difference(self._break_cache):
                    text = self.decode([token_id])
                    self._break_cache[token_id] = any(
                        character in text for character in ".!?\n"
                    )
        return [self._break_cache[token_id] for token_id in token_ids]


class _IncrementalChatDecoder:
    def __init__(self, codec: _OpenAIChatCodec) -> None:
        self.codec = codec
        self.token_cache: list[int] = []
        self.printed_length = 0

    def push(self, token_id: int) -> str:
        self.token_cache.append(int(token_id))
        text = self.codec.decode(self.token_cache)
        if text.endswith("\n"):
            printable = text[self.printed_length :]
            self.token_cache.clear()
            self.printed_length = 0
            return printable
        if text and _is_cjk_character(text[-1]):
            printable = text[self.printed_length :]
            self.printed_length += len(printable)
            return printable
        printable = text[self.printed_length : text.rfind(" ") + 1]
        self.printed_length += len(printable)
        return printable

    def finish(self) -> str:
        text = self.codec.decode(self.token_cache)
        printable = text[self.printed_length :]
        self.token_cache.clear()
        self.printed_length = 0
        return printable


def _is_cjk_character(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x4E00 <= codepoint <= 0x9FFF
        or 0x3400 <= codepoint <= 0x4DBF
        or 0x20000 <= codepoint <= 0x2A6DF
        or 0x2A700 <= codepoint <= 0x2B73F
        or 0x2B740 <= codepoint <= 0x2B81F
        or 0x2B820 <= codepoint <= 0x2CEAF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x2F800 <= codepoint <= 0x2FA1F
    )


def _openai_chat_messages(body: dict[str, Any]) -> list[dict[str, str]]:
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    normalized: list[dict[str, str]] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("each message must be an object")
        role = message.get("role")
        if role not in {"system", "developer", "user", "assistant"}:
            raise ValueError(f"unsupported chat role {role!r}")
        content = message.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if (
                    not isinstance(part, dict)
                    or part.get("type") not in {"text", "input_text"}
                    or not isinstance(part.get("text"), str)
                ):
                    raise ValueError("only text chat content is supported")
                parts.append(part["text"])
            text = "".join(parts)
        else:
            raise ValueError("message content must be text")
        normalized.append({"role": role, "content": text})
    return normalized


def _openai_chat_request(
    body: dict[str, Any],
    *,
    codec: _OpenAIChatCodec,
    default_model: str,
    request_id: str | None,
    openwebui_user_id: str | None,
    openwebui_chat_id: str | None,
    stateful_chat_contract: str | None,
    assistant_message_id: str | None,
    user_message_id: str | None,
    parent_message_id: str | None,
    server_ignore_eos: bool,
) -> dict[str, Any]:
    if body.get("tools") not in (None, []):
        raise ValueError("tools are not supported")
    if body.get("tool_choice") not in (None, "none"):
        raise ValueError("tool_choice is not supported")
    if body.get("stop") not in (None, [], ""):
        raise ValueError("per-request stop sequences are not supported")
    if int(body.get("n", 1)) != 1:
        raise ValueError("only n=1 is supported")
    temperature = body.get("temperature", 0.0)
    if temperature is not None and float(temperature) != 0.0:
        raise ValueError("only greedy temperature=0 is supported")
    top_p = body.get("top_p", 1.0)
    if top_p is not None and float(top_p) != 1.0:
        raise ValueError("only top_p=1 is supported")
    if body.get("logprobs") not in (None, False):
        raise ValueError("logprobs are not supported")
    ignore_eos = _bool_field(body, "ignore_eos", server_ignore_eos)
    if ignore_eos != server_ignore_eos:
        raise ValueError(
            "per-request ignore_eos must match the server EOS policy; "
            "restart with or without --ignore-eos"
        )

    raw_max_tokens = body.get("max_tokens", body.get("max_completion_tokens", 256))
    max_tokens = int(raw_max_tokens)
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    stream_options = body.get("stream_options") or {}
    if not isinstance(stream_options, dict):
        raise ValueError("stream_options must be an object")
    include_usage = stream_options.get("include_usage", False)
    if not isinstance(include_usage, bool):
        raise ValueError("stream_options.include_usage must be a boolean")
    timeout_s = body.get("timeout_s")
    if timeout_s is not None:
        timeout_s = float(timeout_s)

    messages = _openai_chat_messages(body)
    prompt_ids = codec.prompt_token_ids(messages)
    model = str(body.get("model") or default_model)
    explicit_session_id = body.get("session_id")
    if explicit_session_id is not None:
        openwebui_chat_id = str(explicit_session_id)
    chat_id = _chat_identity(openwebui_chat_id, "Open WebUI chat ID")
    user_id = _chat_identity(openwebui_user_id, "Open WebUI user ID")
    contract = _chat_identity(stateful_chat_contract, "stateful chat contract")
    assistant_id = _chat_identity(
        assistant_message_id,
        "assistant message ID",
    )
    user_message = _chat_identity(user_message_id, "user message ID")
    parent_id = _chat_identity(parent_message_id, "parent message ID")
    if contract is not None:
        if contract != PARENT_TOKEN_CHAT_CONTRACT:
            raise ValueError(f"unsupported stateful chat contract {contract!r}")
        if chat_id is None or user_id is None:
            raise ValueError(
                "parent-token-v1 requires Open WebUI user and chat identity headers"
            )
        if assistant_id is None or user_message is None:
            raise ValueError(
                "parent-token-v1 requires assistant and user message ID headers"
            )
    elif any(value is not None for value in (assistant_id, user_message, parent_id)):
        raise ValueError(
            "WKVM message identity headers require X-WKVM-Stateful-Chat"
        )
    session_id = None
    if chat_id is not None:
        # Open WebUI chat IDs are only unique within a user account. Keep the
        # chat-only form as a fallback for older clients that do not forward
        # user headers, while isolating identical chat IDs across users.
        session_id = json.dumps(
            [model, user_id, chat_id],
            ensure_ascii=True,
            separators=(",", ":"),
        )
    response_id = str(
        body.get("request_id")
        or body.get("req_id")
        or request_id
        or f"chatcmpl-{time.time_ns()}"
    )
    return {
        "model": model,
        "messages": messages,
        "prompt_ids": prompt_ids,
        "break_mask": codec.break_mask(prompt_ids),
        "max_tokens": max_tokens,
        "stream": _bool_field(body, "stream", False),
        "include_usage": include_usage,
        "req_id": response_id,
        "session_id": session_id,
        "parent_bound_contract": contract,
        "assistant_message_id": assistant_id,
        "user_message_id": user_message,
        "parent_message_id": parent_id,
        "timeout_s": timeout_s,
    }


def _openai_chat_response(
    *,
    req_id: str,
    model: str,
    text: str,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str | None,
) -> dict[str, Any]:
    return {
        "id": req_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "logprobs": None,
                "finish_reason": _openai_finish_reason(finish_reason),
            }
        ],
        "usage": _openai_usage(prompt_tokens, completion_tokens),
    }


def _openai_chat_chunk(
    *,
    req_id: str,
    model: str,
    delta: dict[str, str],
    finish_reason: str | None = None,
) -> dict[str, Any]:
    return {
        "id": req_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "logprobs": None,
                "finish_reason": _openai_finish_reason(finish_reason),
            }
        ],
    }


def _openai_error(message: str, code: str = "server_error") -> dict[str, Any]:
    return {"error": {"message": message, "type": code, "code": code}}


class _RequestBodyError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


class _TokenSSEWriter:
    def __init__(self, wfile, *, flush_tokens: int = 1) -> None:
        flush_tokens = int(flush_tokens)
        if flush_tokens < 1:
            raise ValueError("flush_tokens must be >= 1")
        self.wfile = wfile
        self.flush_tokens = flush_tokens
        self.pending_payloads: list[bytes] = []
        self.pending_token_events = 0
        self.first_token_flushed = False

    def send(self, event: dict[str, Any]) -> None:
        self.pending_payloads.append(
            (
                f"event: {event['type']}\n"
                f"data: {json.dumps(event)}\n\n"
            ).encode()
        )
        if event.get("type") != "token":
            self.flush()
            return
        if not self.first_token_flushed:
            self.first_token_flushed = True
            self.flush()
            return
        self.pending_token_events += 1
        if self.pending_token_events >= self.flush_tokens:
            self.flush()

    def flush(self) -> None:
        if not self.pending_payloads:
            return
        payload = (
            self.pending_payloads[0]
            if len(self.pending_payloads) == 1
            else b"".join(self.pending_payloads)
        )
        self.wfile.write(payload)
        self.wfile.flush()
        self.pending_payloads.clear()
        self.pending_token_events = 0


def _write_token_sse_stream(
    wfile,
    stream_iter: Iterator[dict[str, Any]],
    first_event: dict[str, Any],
    *,
    flush_tokens: int,
) -> None:
    writer = _TokenSSEWriter(wfile, flush_tokens=flush_tokens)
    try:
        try:
            event = first_event
            while True:
                writer.send(event)
                event = next(stream_iter)
        except StopIteration:
            writer.flush()
        except TimeoutError as exc:
            writer.send({"type": "error", "error": str(exc)})
    except (BrokenPipeError, ConnectionResetError):
        close_stream = getattr(stream_iter, "close", None)
        if close_stream is not None:
            close_stream()


def build_app(
    service: BoundedGemmaService,
    *,
    tokenizer=None,
    model_id: str = "wkvm-gemma",
    ignore_eos: bool | None = None,
    max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES,
    request_read_timeout_s: float = DEFAULT_REQUEST_READ_TIMEOUT_S,
    stream_flush_tokens: int = 1,
):
    if max_request_body_bytes < 1:
        raise ValueError("max_request_body_bytes must be >= 1")
    request_read_timeout_s = float(request_read_timeout_s)
    if not math.isfinite(request_read_timeout_s) or request_read_timeout_s <= 0:
        raise ValueError("request_read_timeout_s must be finite and > 0")
    stream_flush_tokens = int(stream_flush_tokens)
    if stream_flush_tokens < 1:
        raise ValueError("stream_flush_tokens must be >= 1")
    model_id = str(model_id).strip()
    if not model_id:
        raise ValueError("model_id must not be empty")
    model_created = int(time.time())
    chat_codec = None if tokenizer is None else _OpenAIChatCodec(tokenizer)
    if ignore_eos is None:
        ignore_eos = tokenizer is None

    class Handler(BaseHTTPRequestHandler):
        def setup(self) -> None:
            super().setup()
            self.connection.settimeout(request_read_timeout_s)

        def _json(self, code: int, payload) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            if self.headers.get_all("Transfer-Encoding", []):
                raise _RequestBodyError(
                    HTTPStatus.BAD_REQUEST,
                    "Transfer-Encoding is not supported; use Content-Length",
                )
            content_lengths = self.headers.get_all("Content-Length", [])
            if not content_lengths:
                raise _RequestBodyError(
                    HTTPStatus.LENGTH_REQUIRED,
                    "Content-Length header is required",
                )
            if len(content_lengths) != 1:
                raise _RequestBodyError(
                    HTTPStatus.BAD_REQUEST,
                    "exactly one Content-Length header is required",
                )
            raw_content_length = content_lengths[0].strip()
            if not raw_content_length.isdecimal():
                raise _RequestBodyError(
                    HTTPStatus.BAD_REQUEST,
                    "Content-Length header must be a non-negative integer",
                )
            content_length = int(raw_content_length)
            if content_length > max_request_body_bytes:
                raise _RequestBodyError(
                    HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    f"request body exceeds {max_request_body_bytes} bytes",
                )

            deadline = time.monotonic() + request_read_timeout_s
            body = bytearray()
            read = getattr(self.rfile, "read1", self.rfile.read)
            try:
                while len(body) < content_length:
                    remaining_s = deadline - time.monotonic()
                    if remaining_s <= 0:
                        raise TimeoutError
                    self.connection.settimeout(remaining_s)
                    chunk = read(
                        min(
                            content_length - len(body),
                            _REQUEST_BODY_READ_CHUNK_BYTES,
                        )
                    )
                    if not chunk:
                        raise _RequestBodyError(
                            HTTPStatus.BAD_REQUEST,
                            "request body ended before Content-Length bytes were received",
                        )
                    body.extend(chunk)
            except TimeoutError as exc:
                raise _RequestBodyError(
                    HTTPStatus.REQUEST_TIMEOUT,
                    (
                        "request body was not received within "
                        f"{request_read_timeout_s:g} seconds"
                    ),
                ) from exc
            finally:
                self.connection.settimeout(request_read_timeout_s)

            payload = json.loads(body or b"{}")
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            return payload

        def _sse_event(self, event: dict[str, Any] | str) -> None:
            if isinstance(event, str):
                payload = f"data: {event}\n\n".encode()
            else:
                payload = f"data: {json.dumps(event)}\n\n".encode()
            self.wfile.write(payload)
            self.wfile.flush()

        def log_message(self, *a) -> None:
            pass

        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/health":
                health = service.health()
                self._json(200 if health["ok"] else 503, health)
            elif path == "/metrics":
                self._json(200, service.metrics())
            elif path == "/v1/models":
                self._json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": model_id,
                                "object": "model",
                                "created": model_created,
                                "owned_by": "wkvm",
                            }
                        ],
                    },
                )
            elif path.startswith("/v1/status/"):
                req_id = unquote(path.removeprefix("/v1/status/"))
                try:
                    self._json(200, service.status(req_id))
                except KeyError as exc:
                    self._json(404, {"error": str(exc)})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            try:
                body = self._body()
                if path == "/v1/generate":
                    prompt_ids = list(body["prompt_ids"])
                    max_tokens = int(body.get("max_tokens", 32))
                    timeout_s = body.get("timeout_s")
                    if timeout_s is not None:
                        timeout_s = float(timeout_s)
                    break_mask = body.get("break_mask")
                    if break_mask is not None:
                        break_mask = [bool(x) for x in break_mask]
                    self._json(
                        200,
                        service.generate(
                            prompt_ids=prompt_ids,
                            max_tokens=max_tokens,
                            req_id=body.get("req_id"),
                            break_mask=break_mask,
                            timeout_s=timeout_s,
                        ),
                    )
                elif path == "/v1/stream":
                    max_tokens = int(body.get("max_tokens", 32))
                    timeout_s = body.get("timeout_s")
                    if timeout_s is not None:
                        timeout_s = float(timeout_s)
                    break_mask = body.get("break_mask")
                    if break_mask is not None:
                        break_mask = [bool(x) for x in break_mask]
                    req_id = body.get("req_id") or f"gemma-stream-{time.time_ns()}"
                    session_id = body.get("session_id")
                    delta_fields = [
                        name
                        for name in ("delta_ids", "continuation_ids")
                        if name in body
                    ]
                    if len(delta_fields) > 1:
                        raise ValueError(
                            "provide only one of delta_ids or continuation_ids"
                        )
                    if session_id is None:
                        if delta_fields:
                            raise ValueError(
                                "continuation token ids require an explicit session_id"
                            )
                        if "forced_output_ids" in body:
                            raise ValueError(
                                "forced_output_ids require a stateful token session"
                            )
                        stream_iter = service.stream(
                            prompt_ids=list(body["prompt_ids"]),
                            max_tokens=max_tokens,
                            req_id=req_id,
                            break_mask=break_mask,
                            timeout_s=timeout_s,
                        )
                    else:
                        has_prompt = "prompt_ids" in body
                        if int(has_prompt) + len(delta_fields) != 1:
                            raise ValueError(
                                "stateful token request requires exactly one of "
                                "prompt_ids, delta_ids, or continuation_ids"
                            )
                        prompt_ids = (
                            _token_id_list_field(body, "prompt_ids")
                            if has_prompt
                            else None
                        )
                        delta_ids = (
                            _token_id_list_field(body, delta_fields[0])
                            if delta_fields
                            else None
                        )
                        forced_output_ids = (
                            _token_id_list_field(body, "forced_output_ids")
                            if "forced_output_ids" in body
                            else None
                        )
                        stream_iter = service.stream_token_session(
                            session_id=str(session_id),
                            prompt_ids=prompt_ids,
                            delta_ids=delta_ids,
                            forced_output_ids=forced_output_ids,
                            max_tokens=max_tokens,
                            req_id=str(req_id),
                            break_mask=break_mask,
                            timeout_s=timeout_s,
                        )
                    first_event = next(stream_iter)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    _write_token_sse_stream(
                        self.wfile,
                        stream_iter,
                        first_event,
                        flush_tokens=stream_flush_tokens,
                    )
                elif path == "/v1/submit":
                    prompt_ids = list(body["prompt_ids"])
                    max_tokens = int(body.get("max_tokens", 32))
                    timeout_s = body.get("timeout_s")
                    if timeout_s is not None:
                        timeout_s = float(timeout_s)
                    break_mask = body.get("break_mask")
                    if break_mask is not None:
                        break_mask = [bool(x) for x in break_mask]
                    self._json(
                        202,
                        service.submit(
                            prompt_ids=prompt_ids,
                            max_tokens=max_tokens,
                            req_id=body.get("req_id"),
                            break_mask=break_mask,
                            timeout_s=timeout_s,
                        ),
                    )
                elif path == "/v1/cancel":
                    req_id = str(body["req_id"])
                    try:
                        result = service.cancel(req_id)
                    except KeyError as exc:
                        self._json(404, {"error": str(exc)})
                    else:
                        self._json(200, result)
                elif path == "/v1/completions":
                    self._handle_openai_completion(body)
                elif path == "/v1/chat/completions":
                    self._handle_openai_chat(body)
                else:
                    self._json(404, {"error": "not found"})
            except _RequestBodyError as exc:
                self.close_connection = True
                self._json(int(exc.status), {"error": str(exc)})
            except ServiceUnavailable as exc:
                self._json(503, {"error": str(exc)})
            except QueueFull as exc:
                self._json(429, {"error": str(exc)})
            except SessionBusy as exc:
                self._json(409, _openai_error(str(exc), code="session_busy"))
            except TimeoutError as exc:
                self._json(504, {"error": str(exc)})
            except (KeyError, ValueError) as exc:
                self._json(400, {"error": str(exc)})
            except Exception as exc:
                service.total_errors += 1
                self._json(500, {"error": str(exc)})

        def _handle_openai_completion(self, body: dict[str, Any]) -> None:
            req = _openai_completion_request(
                body,
                self.headers.get("x-request-id"),
                server_ignore_eos=ignore_eos,
            )
            if not req["stream"]:
                out = service.generate(
                    prompt_ids=req["prompt_ids"],
                    max_tokens=req["max_tokens"],
                    req_id=req["req_id"],
                    timeout_s=req["timeout_s"],
                )
                if out.get("error"):
                    self._json(500, _openai_error(str(out["error"])))
                    return
                self._json(
                    200,
                    _openai_completion_response(
                        req_id=out["req_id"],
                        model=req["model"],
                        prompt_tokens=len(req["prompt_ids"]),
                        output_tokens=list(out["tokens"]),
                        finish_reason=out.get("finish_reason"),
                        return_token_ids=req["return_token_ids"],
                    ),
                )
                return

            stream_iter = service.stream(
                prompt_ids=req["prompt_ids"],
                max_tokens=req["max_tokens"],
                req_id=req["req_id"],
                timeout_s=req["timeout_s"],
            )
            first_event = next(stream_iter)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            emitted = 0
            try:
                event = first_event
                while True:
                    event_type = event.get("type")
                    if event_type == "queued":
                        event = next(stream_iter)
                        continue
                    if event_type == "token":
                        emitted += 1
                        self._sse_event(
                            {
                                "id": req["req_id"],
                                "object": "text_completion",
                                "created": int(time.time()),
                                "model": req["model"],
                                "choices": [
                                    _openai_completion_choice(
                                        token_ids=[int(event["token"])],
                                        return_token_ids=req["return_token_ids"],
                                    )
                                ],
                            }
                        )
                        event = next(stream_iter)
                    elif event_type == "finish":
                        finish_reason = event.get("finish_reason")
                        error = event.get("error")
                        if error:
                            self._sse_event(_openai_error(str(error)))
                            break
                        self._sse_event(
                            {
                                "id": req["req_id"],
                                "object": "text_completion",
                                "created": int(time.time()),
                                "model": req["model"],
                                "choices": [
                                    _openai_completion_choice(
                                        finish_reason=finish_reason,
                                        return_token_ids=req["return_token_ids"],
                                    )
                                ],
                            }
                        )
                        if req["include_usage"]:
                            self._sse_event(
                                {
                                    "id": req["req_id"],
                                    "object": "text_completion",
                                    "created": int(time.time()),
                                    "model": req["model"],
                                    "choices": [],
                                    "usage": _openai_usage(len(req["prompt_ids"]), emitted),
                                }
                            )
                        break
                self._sse_event("[DONE]")
            except TimeoutError as exc:
                self._sse_event(_openai_error(str(exc), code="timeout"))
                self._sse_event("[DONE]")
            except (BrokenPipeError, ConnectionResetError):
                service.cancel(str(req["req_id"]))

        def _handle_openai_chat(self, body: dict[str, Any]) -> None:
            if chat_codec is None:
                raise ServiceUnavailable("chat completions require a tokenizer")
            req = _openai_chat_request(
                body,
                codec=chat_codec,
                default_model=model_id,
                request_id=self.headers.get("x-request-id"),
                openwebui_user_id=self.headers.get("x-openwebui-user-id"),
                openwebui_chat_id=self.headers.get("x-openwebui-chat-id"),
                stateful_chat_contract=self.headers.get(
                    "x-wkvm-stateful-chat"
                ),
                assistant_message_id=self.headers.get(
                    "x-wkvm-assistant-message-id"
                ),
                user_message_id=self.headers.get("x-wkvm-user-message-id"),
                parent_message_id=self.headers.get("x-wkvm-parent-message-id"),
                server_ignore_eos=ignore_eos,
            )
            if not req["stream"]:
                if req["session_id"] is None:
                    out = service.generate(
                        prompt_ids=req["prompt_ids"],
                        max_tokens=req["max_tokens"],
                        req_id=req["req_id"],
                        break_mask=req["break_mask"],
                        timeout_s=req["timeout_s"],
                    )
                else:
                    out = service.generate_chat(
                        prompt_ids=req["prompt_ids"],
                        max_tokens=req["max_tokens"],
                        session_id=req["session_id"],
                        req_id=req["req_id"],
                        break_mask=req["break_mask"],
                        timeout_s=req["timeout_s"],
                        chat_messages=req["messages"],
                        chat_codec=chat_codec,
                        parent_bound_contract=req["parent_bound_contract"],
                        assistant_message_id=req["assistant_message_id"],
                        user_message_id=req["user_message_id"],
                        parent_message_id=req["parent_message_id"],
                    )
                if out.get("error"):
                    self._json(500, _openai_error(str(out["error"])))
                    return
                output_tokens = list(out["tokens"])
                visible_text = chat_codec.decode(output_tokens)
                if (
                    req["session_id"] is not None
                    and req["parent_bound_contract"] is not None
                ):
                    service.commit_chat_visible_output(
                        session_id=req["session_id"],
                        response_id=req["req_id"],
                        text=visible_text,
                    )
                self._json(
                    200,
                    _openai_chat_response(
                        req_id=req["req_id"],
                        model=req["model"],
                        text=visible_text,
                        prompt_tokens=len(req["prompt_ids"]),
                        completion_tokens=len(output_tokens),
                        finish_reason=out.get("finish_reason"),
                    ),
                )
                return

            if req["session_id"] is None:
                stream_iter = service.stream(
                    prompt_ids=req["prompt_ids"],
                    max_tokens=req["max_tokens"],
                    req_id=req["req_id"],
                    break_mask=req["break_mask"],
                    timeout_s=req["timeout_s"],
                )
            else:
                stream_iter = service.stream_chat(
                    prompt_ids=req["prompt_ids"],
                    max_tokens=req["max_tokens"],
                    session_id=req["session_id"],
                    req_id=req["req_id"],
                    break_mask=req["break_mask"],
                    timeout_s=req["timeout_s"],
                    chat_messages=req["messages"],
                    chat_codec=chat_codec,
                    parent_bound_contract=req["parent_bound_contract"],
                    assistant_message_id=req["assistant_message_id"],
                    user_message_id=req["user_message_id"],
                    parent_message_id=req["parent_message_id"],
                )
            first_event = next(stream_iter)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            decoder = _IncrementalChatDecoder(chat_codec)
            emitted = 0
            visible_parts: list[str] = []
            try:
                self._sse_event(
                    _openai_chat_chunk(
                        req_id=req["req_id"],
                        model=req["model"],
                        delta={"role": "assistant", "content": ""},
                    )
                )
                event = first_event
                while True:
                    event_type = event.get("type")
                    if event_type == "queued":
                        event = next(stream_iter)
                        continue
                    if event_type == "token":
                        emitted += 1
                        text_delta = decoder.push(int(event["token"]))
                        if text_delta:
                            self._sse_event(
                                _openai_chat_chunk(
                                    req_id=req["req_id"],
                                    model=req["model"],
                                    delta={"content": text_delta},
                                )
                            )
                            visible_parts.append(text_delta)
                        event = next(stream_iter)
                        continue
                    if event_type != "finish":
                        event = next(stream_iter)
                        continue
                    final_text = decoder.finish()
                    if final_text:
                        self._sse_event(
                            _openai_chat_chunk(
                                req_id=req["req_id"],
                                model=req["model"],
                                delta={"content": final_text},
                            )
                        )
                        visible_parts.append(final_text)
                    error = event.get("error")
                    if error:
                        self._sse_event(_openai_error(str(error)))
                        break
                    if (
                        req["session_id"] is not None
                        and req["parent_bound_contract"] is not None
                    ):
                        service.commit_chat_visible_output(
                            session_id=req["session_id"],
                            response_id=req["req_id"],
                            text="".join(visible_parts),
                        )
                    self._sse_event(
                        _openai_chat_chunk(
                            req_id=req["req_id"],
                            model=req["model"],
                            delta={},
                            finish_reason=event.get("finish_reason"),
                        )
                    )
                    if req["include_usage"]:
                        self._sse_event(
                            {
                                "id": req["req_id"],
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": req["model"],
                                "choices": [],
                                "usage": _openai_usage(
                                    len(req["prompt_ids"]), emitted
                                ),
                            }
                        )
                    break
                self._sse_event("[DONE]")
            except TimeoutError as exc:
                self._sse_event(_openai_error(str(exc), code="timeout"))
                self._sse_event("[DONE]")
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                close_stream = getattr(stream_iter, "close", None)
                if close_stream is not None:
                    close_stream()

    return Handler


class _GemmaThreadingHTTPServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address,
        request_handler_class,
        *,
        request_queue_size: int,
    ) -> None:
        self.request_queue_size = int(request_queue_size)
        super().__init__(server_address, request_handler_class)


def serve(
    service: BoundedGemmaService,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    tokenizer=None,
    model_id: str = "wkvm-gemma",
    ignore_eos: bool | None = None,
    max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES,
    request_read_timeout_s: float = DEFAULT_REQUEST_READ_TIMEOUT_S,
    stream_flush_tokens: int = 1,
):
    server = _GemmaThreadingHTTPServer(
        (host, port),
        build_app(
            service,
            tokenizer=tokenizer,
            model_id=model_id,
            ignore_eos=ignore_eos,
            max_request_body_bytes=max_request_body_bytes,
            request_read_timeout_s=request_read_timeout_s,
            stream_flush_tokens=stream_flush_tokens,
        ),
        request_queue_size=max(
            int(service.max_queue),
            int(ThreadingHTTPServer.request_queue_size),
        ),
    )
    return server


def run_server(service: BoundedGemmaService, server) -> None:
    install_sigterm_handler = threading.current_thread() is threading.main_thread()
    previous_sigterm_handler = None

    def terminate(_signum, _frame) -> None:
        raise KeyboardInterrupt

    if install_sigterm_handler:
        previous_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, terminate)
    try:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    finally:
        if install_sigterm_handler:
            signal.signal(signal.SIGTERM, previous_sigterm_handler)
        try:
            service.close()
        finally:
            server.server_close()


def apply_native_gemma_production_profile(args) -> None:
    if not getattr(args, "native_gemma_production_profile", False):
        return
    args.native_gemma_checkpoint_loader = True
    args.use_native_gemma_forward = True
    args.native_gemma_attention_backend = "sdpa_single_gqa"
    args.native_gemma_projection_backend = "separate"
    args.native_gemma_weight_backend = "hf_live"
    args.native_gemma_release_hf_decoder_layers = False
    args.persistent_padded_decode_cuda_graph = False
    args.persistent_padded_decode_steps = 128
    args.sink = 16
    args.window = 1024
    args.m_slots = 64
    args.route_chunk = 512
    args.attn = "eager"


def validate_native_gemma_loader_args(args) -> None:
    if not getattr(args, "native_gemma_checkpoint_loader", False):
        return
    args.use_native_gemma_forward = True
    if args.native_gemma_weight_backend != "hf_live":
        raise ValueError(
            "--native-gemma-checkpoint-loader owns checkpoint tensors directly "
            "and requires --native-gemma-weight-backend hf_live"
        )
    if args.native_gemma_release_hf_decoder_layers:
        raise ValueError(
            "--native-gemma-checkpoint-loader does not construct HF decoder "
            "layers, so --native-gemma-release-hf-decoder-layers is invalid"
        )


def engine_kwargs_from_args(args) -> dict[str, Any]:
    return {
        "prefill_chunk": getattr(args, "prefill_chunk", 2048),
        "prefill_microbatch_rows": getattr(args, "prefill_microbatch_rows", 1),
        "continuation_prefill_microbatch_rows": getattr(
            args,
            "continuation_prefill_microbatch_rows",
            None,
        ),
        "decode_microbatch_rows": args.decode_microbatch_rows,
        "decode_microbatch_bytes": args.decode_microbatch_bytes,
        "decode_batch_planner": args.decode_batch_planner,
        "decode_workspace_bytes": args.decode_workspace_bytes,
        "decode_workspace_width_bucket": args.decode_workspace_width_bucket,
        "persistent_exact_decode": not args.disable_persistent_exact_decode,
        "persistent_padded_decode": not args.disable_persistent_padded_decode,
        "persistent_padded_decode_steps": args.persistent_padded_decode_steps,
        "persistent_padded_full_attention_rows": getattr(
            args,
            "persistent_padded_full_attention_rows",
            None,
        ),
        "persistent_padded_decode_cuda_graph": args.persistent_padded_decode_cuda_graph,
        "persistent_padded_decode_graph_warmup_iters": (
            args.persistent_padded_decode_graph_warmup_iters
        ),
        "persistent_padded_sliding_metadata_padding": getattr(
            args,
            "persistent_padded_sliding_metadata_padding",
            False,
        ),
        "use_native_gemma_forward": args.use_native_gemma_forward,
        "native_gemma_attention_backend": args.native_gemma_attention_backend,
        "native_gemma_projection_backend": args.native_gemma_projection_backend,
        "native_gemma_weight_backend": args.native_gemma_weight_backend,
        "native_gemma_release_hf_decoder_layers": (
            args.native_gemma_release_hf_decoder_layers
        ),
        "native_gemma_kv_sharing_fast_prefill": getattr(
            args,
            "native_gemma_kv_sharing_fast_prefill",
            False,
        ),
        "batched_routed_packets": getattr(args, "batched_routed_packets", False),
        "routed_packet_workspace_bytes": getattr(
            args,
            "routed_packet_workspace_bytes",
            64 * 1024 * 1024,
        ),
        "enable_token_pool_metadata": getattr(
            args,
            "enable_token_pool_metadata",
            None,
        ),
        "enable_token_pool_attention": getattr(
            args,
            "enable_token_pool_attention",
            False,
        ),
        "token_pool_max_context_len": getattr(
            args,
            "token_pool_max_context_len",
            None,
        ),
        "token_pool_capacity": getattr(args, "token_pool_capacity", None),
        "token_pool_paged_block_size": getattr(
            args,
            "token_pool_paged_block_size",
            None,
        ),
        "finished_trace_limit": args.max_completed_requests,
    }


def scheduler_config_from_args(args):
    from wkvm.core.config import SchedulerConfig

    prefill_chunk = int(getattr(args, "prefill_chunk", 2048))
    completion_prefill_lane_size = int(
        getattr(args, "completion_prefill_lane_size", 0)
    )
    slots = int(args.slots)
    return SchedulerConfig(
        max_tokens_per_step=max(
            prefill_chunk * slots,
            slots + completion_prefill_lane_size * prefill_chunk,
        ),
        max_running_requests=slots,
        max_tokens_per_request_per_step=prefill_chunk,
        completion_prefill_lane_size=completion_prefill_lane_size,
    )


def _chat_stop_token_ids(full_config, tokenizer, *, ignore_eos: bool) -> frozenset[int]:
    if tokenizer is None or ignore_eos:
        return frozenset()
    raw_stop_token_ids = getattr(full_config, "eos_token_id", ()) or ()
    if isinstance(raw_stop_token_ids, int):
        raw_stop_token_ids = [raw_stop_token_ids]
    stop_token_ids = {int(token_id) for token_id in raw_stop_token_ids}
    for attribute in ("eos_token_id", "eot_token_id"):
        token_id = getattr(tokenizer, attribute, None)
        if token_id is not None:
            stop_token_ids.add(int(token_id))
    return frozenset(stop_token_ids)


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--served-model-name", default=None)
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--max-queue", type=int, default=64)
    ap.add_argument("--request-timeout-s", type=float, default=None)
    ap.add_argument(
        "--max-request-body-bytes",
        type=int,
        default=DEFAULT_MAX_REQUEST_BODY_BYTES,
        help="Reject HTTP request bodies larger than this many bytes (default: 8 MiB).",
    )
    ap.add_argument(
        "--request-read-timeout-s",
        type=float,
        default=DEFAULT_REQUEST_READ_TIMEOUT_S,
        help="Maximum wall-clock seconds to receive one HTTP request body (default: 30).",
    )
    ap.add_argument(
        "--stream-flush-tokens",
        type=int,
        default=1,
        help=(
            "Token events per /v1/stream socket write after the first token; "
            "queued, first-token, finish, and error events flush immediately."
        ),
    )
    ap.add_argument("--max-completed-requests", type=int, default=4096)
    ap.add_argument(
        "--enable-openai-chat",
        action="store_true",
        help=(
            "Load the tokenizer and enable /v1/chat/completions. Disabled by "
            "default so token-ID serving keeps its original startup path."
        ),
    )
    ap.add_argument(
        "--ignore-eos",
        action="store_true",
        help=(
            "Ignore EOS/EOT globally and generate exactly max_tokens; intended "
            "for fixed-output benchmark runs."
        ),
    )
    ap.add_argument("--chat-session-ttl-s", type=float, default=1800.0)
    ap.add_argument("--max-chat-sessions", type=int, default=None)
    ap.add_argument(
        "--enable-token-session-teacher-forcing",
        action="store_true",
        help=(
            "Allow stateful /v1/stream requests to replace each sampled pending "
            "token with forced_output_ids for fixed-trace benchmark replay."
        ),
    )
    ap.add_argument(
        "--batch-wait-s",
        type=float,
        default=0.01,
        help="Idle cohort collection delay before the first engine step.",
    )
    ap.add_argument(
        "--empty-cuda-cache-before-decode",
        action="store_true",
        help=(
            "Release inactive CUDA allocator blocks after every chat cohort has "
            "produced its first token and before sustained decode."
        ),
    )
    ap.add_argument(
        "--prefill-chunk",
        type=int,
        default=2048,
        help="Maximum prompt tokens consumed per request in one scheduler step.",
    )
    ap.add_argument(
        "--prefill-microbatch-rows",
        type=int,
        default=1,
        help="Equal-width prompt rows per model call; 0 is unlimited.",
    )
    ap.add_argument(
        "--continuation-prefill-microbatch-rows",
        type=int,
        default=None,
        help="Continuation prompt rows per model call; 0 is unlimited.",
    )
    ap.add_argument("--completion-prefill-lane-size", type=int, default=0)
    ap.add_argument("--decode-microbatch-rows", type=int, default=16)
    ap.add_argument("--decode-microbatch-bytes", type=int, default=None)
    ap.add_argument("--decode-workspace-bytes", type=int, default=None)
    ap.add_argument("--decode-workspace-width-bucket", type=int, default=16)
    ap.add_argument("--disable-persistent-exact-decode", action="store_true")
    ap.add_argument("--disable-persistent-padded-decode", action="store_true")
    ap.add_argument("--persistent-padded-decode-steps", type=int, default=8)
    full_attention_rows_group = ap.add_mutually_exclusive_group()
    full_attention_rows_group.add_argument(
        "--persistent-padded-full-attention-rows",
        dest="persistent_padded_full_attention_rows",
        action="store_true",
        default=None,
    )
    full_attention_rows_group.add_argument(
        "--disable-persistent-padded-full-attention-rows",
        dest="persistent_padded_full_attention_rows",
        action="store_false",
    )
    ap.add_argument("--persistent-padded-decode-cuda-graph", action="store_true")
    ap.add_argument("--persistent-padded-decode-graph-warmup-iters", type=int, default=3)
    ap.add_argument(
        "--persistent-padded-sliding-metadata-padding",
        action="store_true",
        help="Pad sliding token-pool metadata to graph-stable shapes.",
    )
    ap.add_argument(
        "--enable-token-pool-metadata",
        action="store_true",
        default=None,
        help="Build request/token metadata even when token-pool attention is disabled.",
    )
    ap.add_argument(
        "--enable-token-pool-attention",
        action="store_true",
        help="Use the native token-pool attention path; implies native Gemma forward.",
    )
    ap.add_argument("--token-pool-max-context-len", type=int, default=None)
    ap.add_argument("--token-pool-capacity", type=int, default=None)
    ap.add_argument("--token-pool-paged-block-size", type=int, default=None)
    ap.add_argument("--sink", type=int, default=16)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--m-slots", type=int, default=64)
    ap.add_argument("--route-chunk", type=int, default=512)
    ap.add_argument(
        "--native-gemma-production-profile",
        action="store_true",
        help=(
            "Use the measured checkpoint-native production profile: no HF model "
            "construction, native SDPA with single-row GQA, separate projections, "
            "persistent padded decode without CUDA graphs, route_chunk=512, and "
            "128 reserved decode steps."
        ),
    )
    ap.add_argument(
        "--native-gemma-checkpoint-loader",
        action="store_true",
        help=(
            "Load Gemma4 text checkpoint tensors directly into wkvm's native "
            "Gemma facade instead of constructing Gemma4ForCausalLM.from_pretrained."
        ),
    )
    ap.add_argument(
        "--use-native-gemma-forward",
        action="store_true",
        help=(
            "Run model calls through wkvm's NativeGemma4ForCausalLM bridge instead "
            "of transformers.Gemma4ForCausalLM.forward. Still uses loaded HF weights."
        ),
    )
    ap.add_argument(
        "--native-gemma-attention-backend",
        choices=[
            "manual",
            "manual_gqa",
            "sdpa",
            "sdpa_single_gqa",
            "triton_dense_gqa",
        ],
        default="manual",
        help="Attention primitive used inside --use-native-gemma-forward.",
    )
    ap.add_argument(
        "--native-gemma-projection-backend",
        choices=["separate", "qkv_packed", "gate_up_packed", "qkv_gate_up_packed"],
        default="separate",
        help="Projection primitive used inside --use-native-gemma-forward.",
    )
    ap.add_argument(
        "--native-gemma-weight-backend",
        choices=["hf_live", "owned", "owned_cpu"],
        default="hf_live",
        help=(
            "Weight source used inside --use-native-gemma-forward. 'owned' copies "
            "decoder-layer weights into native tensors at bridge construction; "
            "'owned_cpu' keeps those snapshots on CPU and stages per operation."
        ),
    )
    ap.add_argument(
        "--native-gemma-release-hf-decoder-layers",
        action="store_true",
        help=(
            "After constructing the native owned-weight bridge, replace HF decoder "
            "layers with empty modules so serving does not keep duplicate decoder "
            "weights resident. Requires --native-gemma-weight-backend owned or "
            "owned_cpu."
        ),
    )
    ap.add_argument(
        "--native-gemma-kv-sharing-fast-prefill",
        action="store_true",
        help=(
            "For Gemma4 KV-sharing models, run the KV-owning prefix layers on "
            "all prompt tokens and the KV-shared tail only at requested logit "
            "positions. Prompt-token logits outside those positions are not valid."
        ),
    )
    ap.add_argument("--batched-routed-packets", action="store_true")
    ap.add_argument(
        "--routed-packet-workspace-bytes",
        type=int,
        default=64 * 1024 * 1024,
    )
    ap.add_argument(
        "--decode-batch-planner",
        choices=["scheduler", "length_bucketed"],
        default="scheduler",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--attn", choices=["eager", "sdpa"], default="sdpa")
    args = ap.parse_args()

    if args.max_request_body_bytes < 1:
        ap.error("--max-request-body-bytes must be >= 1")
    if args.request_timeout_s is not None and (
        not math.isfinite(args.request_timeout_s) or args.request_timeout_s <= 0
    ):
        ap.error("--request-timeout-s must be finite and > 0")
    if (
        args.chat_session_ttl_s is not None
        and (
            not math.isfinite(args.chat_session_ttl_s)
            or args.chat_session_ttl_s <= 0
        )
    ):
        ap.error("--chat-session-ttl-s must be finite and > 0")
    if args.max_chat_sessions is not None and args.max_chat_sessions < 1:
        ap.error("--max-chat-sessions must be >= 1")
    if args.prefill_chunk < 1:
        ap.error("--prefill-chunk must be >= 1")
    if (
        args.continuation_prefill_microbatch_rows is not None
        and args.continuation_prefill_microbatch_rows < 0
    ):
        ap.error("--continuation-prefill-microbatch-rows must be >= 0 or omitted")
    if not math.isfinite(args.batch_wait_s) or args.batch_wait_s < 0:
        ap.error("--batch-wait-s must be finite and >= 0")
    if (
        not math.isfinite(args.request_read_timeout_s)
        or args.request_read_timeout_s <= 0
    ):
        ap.error("--request-read-timeout-s must be finite and > 0")
    if args.stream_flush_tokens < 1:
        ap.error("--stream-flush-tokens must be >= 1")
    for option, value in (
        ("--token-pool-max-context-len", args.token_pool_max_context_len),
        ("--token-pool-capacity", args.token_pool_capacity),
        ("--token-pool-paged-block-size", args.token_pool_paged_block_size),
    ):
        if value is not None and value < 1:
            ap.error(f"{option} must be >= 1")

    try:
        import torch
        from transformers import AutoConfig
        from transformers.models.gemma4 import Gemma4ForCausalLM

        if args.enable_openai_chat:
            from transformers import AutoTokenizer
    except ImportError as exc:
        ap.error(
            "Gemma serving dependencies are unavailable; install "
            "wkvm[gemma-server] (or .[gemma-server] from a checkout): "
            f"{exc}"
        )

    from wkvm.gemma_engine import GemmaNativeEngine
    from wkvm.models.gemma import gemma4_e4b_routed_span_config

    apply_native_gemma_production_profile(args)
    if args.enable_token_pool_attention:
        args.use_native_gemma_forward = True
    validate_native_gemma_loader_args(args)
    full_cfg = AutoConfig.from_pretrained(args.model)
    tokenizer = (
        AutoTokenizer.from_pretrained(args.model)
        if args.enable_openai_chat
        else None
    )

    if args.native_gemma_checkpoint_loader:
        from wkvm.runner.gemma_native_forward import load_native_gemma4_from_checkpoint

        model = load_native_gemma4_from_checkpoint(
            args.model,
            device=args.device,
            dtype=torch.bfloat16,
            native_attention_backend=args.native_gemma_attention_backend,
            native_projection_backend=args.native_gemma_projection_backend,
        )
    else:
        text_cfg = full_cfg.get_text_config(decoder=True)
        model = Gemma4ForCausalLM.from_pretrained(
            args.model,
            config=text_cfg,
            dtype=torch.bfloat16,
            attn_implementation=args.attn,
            key_mapping={r"^model\.language_model": "model"},
            device_map=args.device,
        )
    model.eval()
    cfg = gemma4_e4b_routed_span_config(
        num_hidden_layers=model.config.num_hidden_layers,
        num_kv_shared_layers=getattr(model.config, "num_kv_shared_layers", 0),
        layer_types=tuple(model.config.layer_types),
        num_kv_heads=getattr(model.config, "num_global_key_value_heads", None)
        or getattr(model.config, "num_key_value_heads", 2),
        head_dim=getattr(model.config, "global_head_dim", None)
        or getattr(model.config, "head_dim", 512),
        sink_tokens=args.sink,
        ring_tokens=args.window,
        routed_slots=args.m_slots,
        pending_tokens=args.route_chunk,
        sliding_window=getattr(model.config, "sliding_window", 1024),
    )
    stop_token_ids = _chat_stop_token_ids(
        full_cfg,
        tokenizer,
        ignore_eos=args.ignore_eos,
    )
    engine = GemmaNativeEngine(
        model,
        cfg,
        num_slots=args.slots,
        stop_token_ids=stop_token_ids,
        scheduler_config=scheduler_config_from_args(args),
        **engine_kwargs_from_args(args),
    )
    service = BoundedGemmaService(
        engine,
        max_queue=args.max_queue,
        batch_wait_s=args.batch_wait_s,
        request_timeout_s=args.request_timeout_s,
        max_completed_requests=args.max_completed_requests,
        chat_session_ttl_s=args.chat_session_ttl_s,
        max_chat_sessions=args.max_chat_sessions,
        cuda_empty_cache=(
            torch.cuda.empty_cache
            if args.empty_cuda_cache_before_decode
            else None
        ),
        enable_token_session_teacher_forcing=(
            args.enable_token_session_teacher_forcing
        ),
    )
    server = serve(
        service,
        port=args.port,
        tokenizer=tokenizer,
        model_id=args.served_model_name or args.model.rstrip("/").rsplit("/", 1)[-1],
        ignore_eos=(tokenizer is None or args.ignore_eos),
        max_request_body_bytes=args.max_request_body_bytes,
        request_read_timeout_s=args.request_read_timeout_s,
        stream_flush_tokens=args.stream_flush_tokens,
    )
    stats = engine.stats()
    print(
        "native Gemma wkvm serving on "
        f"127.0.0.1:{args.port} "
        f"backend={stats.get('model_forward_backend')} "
        f"uses_hf_transformer_forward={stats.get('uses_hf_transformer_forward')} "
        f"uses_hf_model_construction={stats.get('uses_hf_model_construction')}"
    )
    run_server(service, server)


if __name__ == "__main__":
    main()
