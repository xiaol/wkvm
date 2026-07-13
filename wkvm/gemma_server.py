"""Native Gemma token-id HTTP endpoint.

The canonical wkvm endpoint is token-id `/v1/stream`. `/v1/completions` exposes
the same engine through the OpenAI completions streaming shape used by vLLM and
SGLang benchmarks, limited to single-prompt greedy token-id requests.
"""

from __future__ import annotations

import json
import math
import signal
import threading
import time
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterator
from urllib.parse import unquote, urlparse


DEFAULT_MAX_REQUEST_BODY_BYTES = 8 * 1024 * 1024
DEFAULT_REQUEST_READ_TIMEOUT_S = 30.0
_REQUEST_BODY_READ_CHUNK_BYTES = 64 * 1024


class BoundedGemmaService:
    def __init__(
        self,
        engine,
        *,
        max_queue: int = 64,
        batch_wait_s: float = 0.01,
        request_timeout_s: float | None = None,
        max_completed_requests: int | None = 4096,
    ) -> None:
        if max_queue < 1:
            raise ValueError("max_queue must be >= 1")
        if request_timeout_s is not None:
            request_timeout_s = float(request_timeout_s)
            if not math.isfinite(request_timeout_s) or request_timeout_s <= 0:
                raise ValueError("request_timeout_s must be finite and > 0 or None")
        if max_completed_requests is not None and max_completed_requests < 1:
            raise ValueError("max_completed_requests must be >= 1 or None")
        self.engine = engine
        self.max_queue = max_queue
        self.batch_wait_s = max(0.0, float(batch_wait_s))
        self.request_timeout_s = request_timeout_s
        self.max_completed_requests = max_completed_requests
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
        self._requests: dict[str, Any] = {}
        self._deadlines: dict[str, float] = {}
        self._completed_order: deque[str] = deque()
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
            pending = len(self._pending)
            with self.engine_lock:
                waiting = len(self.engine.scheduler.waiting)
                running = len(self.engine.scheduler.running)
                free_state_slots = self.engine.arena.num_free_slots()
            return {
                "ok": ready,
                "queue_depth": pending + waiting,
                "pending_queue_depth": pending,
                "running": running,
                "free_state_slots": free_state_slots,
                "timed_out_requests": self.total_timed_out,
                "worker_alive": worker_alive,
                "last_error": self.last_error,
            }

    def metrics(self) -> dict[str, Any]:
        with self.lock:
            worker_alive = self._worker.is_alive()
            ready = self.ready and not self.closed and worker_alive
            pending = len(self._pending)
            with self.engine_lock:
                engine_stats = self.engine.stats()
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
                while (
                    not self.closed
                    and not self._pending
                    and not self.engine.has_unfinished
                ):
                    self.cv.wait(timeout=0.25)
                if self.closed:
                    return
                if (
                    self.batch_wait_s > 0
                    and not self.engine.has_unfinished
                    and self._pending
                ):
                    self.cv.wait(timeout=self.batch_wait_s)
                    if self.closed:
                        return
                self._expire_deadlines_locked()
                pending = list(self._pending)
                self._pending.clear()
            try:
                with self.engine_lock:
                    for request, break_mask in pending:
                        if request.req_id in self.cancelled:
                            self.cancelled.discard(request.req_id)
                            request.status = type(request.status).FINISHED_ABORTED
                            continue
                        self.engine.add_request(request, break_mask=break_mask)
                    if self.engine.has_unfinished:
                        self.engine.step()
            except Exception as exc:
                with self.cv:
                    self.total_errors += 1
                    self.last_error = str(exc)
                    self._fail_unfinished_locked(str(exc).splitlines()[0])
                    self.cv.notify_all()
            else:
                with self.cv:
                    self._expire_deadlines_locked()
                    self._record_completed_locked()
                    self._trim_completed_locked()
                    self.cv.notify_all()

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
            queued = len(self._pending) + len(self.engine.scheduler.waiting)
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
        with self.engine_lock:
            fail_unfinished = getattr(self.engine, "fail_unfinished", None)
            if callable(fail_unfinished):
                fail_unfinished(error)
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


def _openai_completion_request(body: dict[str, Any], req_id: str | None) -> dict[str, Any]:
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
    ignore_eos = _bool_field(body, "ignore_eos", True)
    if not ignore_eos:
        raise ValueError("ignore_eos=false is not supported by token-id completions")

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


def _openai_error(message: str, code: str = "server_error") -> dict[str, Any]:
    return {"error": {"message": message, "type": code, "code": code}}


class _RequestBodyError(Exception):
    def __init__(self, status: HTTPStatus, message: str) -> None:
        super().__init__(message)
        self.status = status


def build_app(
    service: BoundedGemmaService,
    *,
    max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES,
    request_read_timeout_s: float = DEFAULT_REQUEST_READ_TIMEOUT_S,
):
    if max_request_body_bytes < 1:
        raise ValueError("max_request_body_bytes must be >= 1")
    request_read_timeout_s = float(request_read_timeout_s)
    if not math.isfinite(request_read_timeout_s) or request_read_timeout_s <= 0:
        raise ValueError("request_read_timeout_s must be finite and > 0")

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
                    prompt_ids = list(body["prompt_ids"])
                    max_tokens = int(body.get("max_tokens", 32))
                    timeout_s = body.get("timeout_s")
                    if timeout_s is not None:
                        timeout_s = float(timeout_s)
                    break_mask = body.get("break_mask")
                    if break_mask is not None:
                        break_mask = [bool(x) for x in break_mask]
                    req_id = body.get("req_id") or f"gemma-stream-{time.time_ns()}"
                    stream_iter = service.stream(
                        prompt_ids=prompt_ids,
                        max_tokens=max_tokens,
                        req_id=req_id,
                        break_mask=break_mask,
                        timeout_s=timeout_s,
                    )
                    first_event = next(stream_iter)
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    try:
                        event = first_event
                        while True:
                            payload = (
                                f"event: {event['type']}\n"
                                f"data: {json.dumps(event)}\n\n"
                            ).encode()
                            self.wfile.write(payload)
                            self.wfile.flush()
                            event = next(stream_iter)
                    except StopIteration:
                        pass
                    except (BrokenPipeError, ConnectionResetError):
                        service.cancel(str(req_id))
                    except TimeoutError as exc:
                        payload = (
                            "event: error\n"
                            f"data: {json.dumps({'type': 'error', 'error': str(exc)})}\n\n"
                        ).encode()
                        self.wfile.write(payload)
                        self.wfile.flush()
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
                else:
                    self._json(404, {"error": "not found"})
            except _RequestBodyError as exc:
                self.close_connection = True
                self._json(int(exc.status), {"error": str(exc)})
            except ServiceUnavailable as exc:
                self._json(503, {"error": str(exc)})
            except QueueFull as exc:
                self._json(429, {"error": str(exc)})
            except TimeoutError as exc:
                self._json(504, {"error": str(exc)})
            except (KeyError, ValueError) as exc:
                self._json(400, {"error": str(exc)})
            except Exception as exc:
                service.total_errors += 1
                self._json(500, {"error": str(exc)})

        def _handle_openai_completion(self, body: dict[str, Any]) -> None:
            req = _openai_completion_request(body, self.headers.get("x-request-id"))
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

    return Handler


def serve(
    service: BoundedGemmaService,
    *,
    host: str = "127.0.0.1",
    port: int = 8000,
    max_request_body_bytes: int = DEFAULT_MAX_REQUEST_BODY_BYTES,
    request_read_timeout_s: float = DEFAULT_REQUEST_READ_TIMEOUT_S,
):
    server = ThreadingHTTPServer(
        (host, port),
        build_app(
            service,
            max_request_body_bytes=max_request_body_bytes,
            request_read_timeout_s=request_read_timeout_s,
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
    args.persistent_padded_decode_cuda_graph = True
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
        "decode_microbatch_rows": args.decode_microbatch_rows,
        "decode_microbatch_bytes": args.decode_microbatch_bytes,
        "decode_batch_planner": args.decode_batch_planner,
        "decode_workspace_bytes": args.decode_workspace_bytes,
        "decode_workspace_width_bucket": args.decode_workspace_width_bucket,
        "persistent_exact_decode": not args.disable_persistent_exact_decode,
        "persistent_padded_decode": not args.disable_persistent_padded_decode,
        "persistent_padded_decode_steps": args.persistent_padded_decode_steps,
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


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
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
    ap.add_argument("--max-completed-requests", type=int, default=4096)
    ap.add_argument("--decode-microbatch-rows", type=int, default=16)
    ap.add_argument("--decode-microbatch-bytes", type=int, default=None)
    ap.add_argument("--decode-workspace-bytes", type=int, default=None)
    ap.add_argument("--decode-workspace-width-bucket", type=int, default=16)
    ap.add_argument("--disable-persistent-exact-decode", action="store_true")
    ap.add_argument("--disable-persistent-padded-decode", action="store_true")
    ap.add_argument("--persistent-padded-decode-steps", type=int, default=8)
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
            "persistent padded CUDA graph decode, route_chunk=512, and 128 reserved "
            "decode steps."
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
        choices=["manual", "manual_gqa", "sdpa", "sdpa_single_gqa", "triton_dense_gqa"],
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
        not math.isfinite(args.request_read_timeout_s)
        or args.request_read_timeout_s <= 0
    ):
        ap.error("--request-read-timeout-s must be finite and > 0")
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
    except ImportError as exc:
        ap.error(
            "Gemma serving dependencies are unavailable; install "
            "wkvm[gemma-server] (or .[gemma-server] from a checkout): "
            f"{exc}"
        )

    from wkvm.core.config import SchedulerConfig
    from wkvm.gemma_engine import GemmaNativeEngine
    from wkvm.models.gemma import gemma4_e4b_routed_span_config

    apply_native_gemma_production_profile(args)
    if args.enable_token_pool_attention:
        args.use_native_gemma_forward = True
    validate_native_gemma_loader_args(args)

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
        full_cfg = AutoConfig.from_pretrained(args.model)
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
    engine = GemmaNativeEngine(
        model,
        cfg,
        num_slots=args.slots,
        scheduler_config=SchedulerConfig(
            max_tokens_per_step=8192 * args.slots,
            max_running_requests=args.slots,
            max_tokens_per_request_per_step=8192,
        ),
        **engine_kwargs_from_args(args),
    )
    service = BoundedGemmaService(
        engine,
        max_queue=args.max_queue,
        request_timeout_s=args.request_timeout_s,
        max_completed_requests=args.max_completed_requests,
    )
    server = serve(
        service,
        port=args.port,
        max_request_body_bytes=args.max_request_body_bytes,
        request_read_timeout_s=args.request_read_timeout_s,
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
