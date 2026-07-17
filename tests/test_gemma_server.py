import unittest
import http.client
import io
import signal
import threading
import time
import urllib.error
import urllib.request
import json
import sys
from collections import deque
from types import SimpleNamespace
from unittest.mock import patch

from experiments.wkvm_serving_bench import stream_request_openai_completions
from wkvm.gemma_server import (
    BoundedGemmaService,
    QueueFull,
    ServiceUnavailable,
    _ChatTurn,
    _TokenSSEWriter,
    _chat_stop_token_ids,
    _write_token_sse_stream,
    apply_native_gemma_production_profile,
    engine_kwargs_from_args,
    main,
    run_server,
    scheduler_config_from_args,
    serve,
    validate_native_gemma_loader_args,
)


class FakeQueue:
    def __len__(self):
        return 0


class FakeArena:
    def num_free_slots(self):
        return 1


class FakeScheduler:
    waiting = FakeQueue()
    running = []


class FakeEngine:
    def __init__(self):
        self.scheduler = FakeScheduler()
        self.arena = FakeArena()
        self.finished_traces = {}
        self.added = []

    @property
    def has_unfinished(self):
        return any(not req.status.is_finished for req, _ in self.added)

    def add_request(self, req, *, break_mask=None):
        self.added.append((req, break_mask))

    def step(self):
        req = next(req for req, _ in self.added if not req.status.is_finished)
        req.output_token_ids.append(7)
        req.status = type(req.status).FINISHED_LENGTH

    def abort_request(self, req_id):
        pass

    def stats(self):
        return {"queue_depth": 0}


class StepwiseEngine(FakeEngine):
    def step(self):
        for req, _ in self.added:
            if req.status.is_finished:
                continue
            req.output_token_ids.append(100 + len(req.output_token_ids))
            if len(req.output_token_ids) >= req.max_new_tokens:
                req.status = type(req.status).FINISHED_LENGTH
            return


class NeverFinishEngine(FakeEngine):
    def step(self):
        time.sleep(0.01)

    def abort_request(self, req_id):
        for req, _ in self.added:
            if req.req_id == req_id:
                req.status = type(req.status).FINISHED_ABORTED


class ErrorEngine(FakeEngine):
    def step(self):
        raise RuntimeError("synthetic engine failure")


class FakeChatTokenizer:
    _role_tokens = {
        "system": 8,
        "developer": 9,
        "user": 10,
        "assistant": 12,
    }
    _end_tokens = {
        "system": 11,
        "developer": 11,
        "user": 11,
        "assistant": 13,
    }
    _special_tokens = {2, 8, 9, 10, 11, 12, 13}
    eos_token_id = 1
    eot_token_id = 13

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt,
        tokenize,
        return_dict=False,
    ):
        if not add_generation_prompt or not tokenize or return_dict:
            raise AssertionError("unexpected chat template options")
        token_ids = [2]
        for message in messages:
            role = message["role"]
            token_ids.append(self._role_tokens[role])
            token_ids.extend(ord(character) for character in message["content"])
            token_ids.append(self._end_tokens[role])
        token_ids.append(self._role_tokens["assistant"])
        return token_ids

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ):
        del clean_up_tokenization_spaces
        return "".join(
            chr(token_id)
            for token_id in token_ids
            if not skip_special_tokens or token_id not in self._special_tokens
        )


class FakeSessionArena:
    def __init__(self, engine, num_slots=2):
        self.engine = engine
        self.num_slots = num_slots

    def num_free_slots(self):
        resident = len(self.engine.scheduler.running) + len(self.engine.scheduler.parked)
        return max(0, self.num_slots - resident)


class FakeSessionTrace:
    def __init__(self, req_id, finish_reason="length", error=None):
        self.req_id = req_id
        self.finish_reason = finish_reason
        self.error = error

    def as_dict(self):
        return {
            "req_id": self.req_id,
            "finish_reason": self.finish_reason,
            "error": self.error,
        }


class FakeSessionEngine:
    def __init__(self, num_slots=2):
        self.scheduler = SimpleNamespace(
            waiting=deque(),
            running=[],
            parked={},
            requests={},
            config=SimpleNamespace(max_running_requests=num_slots),
        )
        self.arena = FakeSessionArena(self, num_slots=num_slots)
        self.finished_traces = {}
        self.started = []
        self.continuations = []
        self.closed_sessions = []

    @property
    def has_unfinished(self):
        return bool(self.scheduler.waiting or self.scheduler.running)

    def add_session_request(self, request, *, break_mask=None):
        self.scheduler.requests[request.req_id] = request
        self.scheduler.waiting.append(request)
        self.started.append((request.req_id, list(request.prompt_token_ids), break_mask))

    def continue_session_requests(
        self,
        continuations,
        *,
        max_new_tokens,
        break_masks=None,
    ):
        for req_id, tokens in continuations.items():
            request = self.scheduler.parked.pop(req_id)
            request.prompt_token_ids.extend(request.output_token_ids)
            request.prompt_token_ids.extend(tokens)
            request.output_token_ids.clear()
            request.max_new_tokens = max_new_tokens
            request.status = type(request.status).RUNNING
            request.parked_finish_status = None
            self.scheduler.running.append(request)
            self.continuations.append(
                (req_id, list(tokens), None if break_masks is None else break_masks[req_id])
            )

    def step(self):
        from wkvm.core.request import RequestStatus

        while self.scheduler.waiting and len(self.scheduler.running) < self.arena.num_slots:
            request = self.scheduler.waiting.popleft()
            request.status = RequestStatus.RUNNING
            self.scheduler.running.append(request)
        completed = []
        for request in list(self.scheduler.running):
            output = [65, 13][: request.max_new_tokens]
            request.output_token_ids.extend(output)
            request.parked_finish_status = RequestStatus.FINISHED_LENGTH
            request.status = RequestStatus.PARKED
            self.scheduler.running.remove(request)
            self.scheduler.parked[request.req_id] = request
            self.finished_traces[request.req_id] = FakeSessionTrace(request.req_id)
            completed.append(request)
        return completed

    def close_sessions(self, req_ids):
        from wkvm.core.request import RequestStatus

        closed = []
        for req_id in req_ids:
            request = self.scheduler.parked.pop(req_id)
            request.status = RequestStatus.FINISHED_CLOSED
            request.parked_finish_status = None
            self.closed_sessions.append(req_id)
            closed.append(request)
        return closed

    def abort_request(self, req_id):
        from wkvm.core.request import RequestStatus

        request = self.scheduler.requests.get(req_id)
        if request is None:
            return
        try:
            self.scheduler.waiting.remove(request)
        except ValueError:
            pass
        if request in self.scheduler.running:
            self.scheduler.running.remove(request)
        self.scheduler.parked.pop(req_id, None)
        request.status = RequestStatus.FINISHED_ABORTED

    def fail_unfinished(self, error):
        from wkvm.core.request import RequestStatus

        for request in self.scheduler.requests.values():
            if not request.status.is_finished:
                request.status = RequestStatus.FINISHED_ERROR
                self.finished_traces[request.req_id] = FakeSessionTrace(
                    request.req_id,
                    finish_reason="error",
                    error=error,
                )
        self.scheduler.waiting.clear()
        self.scheduler.running.clear()
        self.scheduler.parked.clear()

    def stats(self):
        return {
            "queue_depth": len(self.scheduler.waiting),
            "parked_sessions": len(self.scheduler.parked),
        }


class NeverFinishSessionEngine(FakeSessionEngine):
    def step(self):
        time.sleep(0.01)
        while self.scheduler.waiting and len(self.scheduler.running) < self.arena.num_slots:
            request = self.scheduler.waiting.popleft()
            request.status = type(request.status).RUNNING
            self.scheduler.running.append(request)
        return []


class FatalAfterStepEngine(FakeEngine):
    def __init__(self):
        super().__init__()
        self.fail_worker = False

    @property
    def has_unfinished(self):
        if self.fail_worker:
            raise RuntimeError("synthetic worker failure")
        return super().has_unfinished

    def step(self):
        self.fail_worker = True


class FinishWhileCancelWaitsEngine(FakeEngine):
    def __init__(self):
        super().__init__()
        self.step_started = threading.Event()
        self.release_step = threading.Event()
        self.abort_calls = 0

    def step(self):
        self.step_started.set()
        if not self.release_step.wait(timeout=2):
            raise RuntimeError("test did not release engine step")
        req = next(req for req, _ in self.added if not req.status.is_finished)
        req.output_token_ids.append(7)
        req.status = type(req.status).FINISHED_LENGTH

    def abort_request(self, req_id):
        self.abort_calls += 1
        super().abort_request(req_id)


class FullQueue(FakeQueue):
    def __len__(self):
        return 1


class FullEngine(FakeEngine):
    def __init__(self):
        super().__init__()
        self.scheduler.waiting = FullQueue()


class SigtermServer:
    def __init__(self):
        self.server_close_calls = 0

    def serve_forever(self):
        handler = signal.getsignal(signal.SIGTERM)
        if not callable(handler):
            raise AssertionError("SIGTERM handler was not installed")
        handler(signal.SIGTERM, None)

    def server_close(self):
        self.server_close_calls += 1


class RecordingWFile:
    def __init__(self, *, fail_on_write: int | None = None):
        self.fail_on_write = fail_on_write
        self.writes = []
        self.flush_calls = 0

    def write(self, payload):
        write_number = len(self.writes) + 1
        if self.fail_on_write == write_number:
            raise BrokenPipeError("synthetic disconnect")
        self.writes.append(payload)

    def flush(self):
        self.flush_calls += 1


class ClosableEventIterator:
    def __init__(self, events):
        self.events = iter(events)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.events)

    def close(self):
        self.closed = True


class TestGemmaServerEngineArgs(unittest.TestCase):
    def test_native_forward_flags_are_passed_to_engine_kwargs(self) -> None:
        args = SimpleNamespace(
            prefill_chunk=2048,
            prefill_microbatch_rows=1,
            continuation_prefill_microbatch_rows=8,
            decode_microbatch_rows=8,
            decode_microbatch_bytes=370_000_000,
            decode_batch_planner="length_bucketed",
            decode_workspace_bytes=470_000_000,
            decode_workspace_width_bucket=32,
            disable_persistent_exact_decode=True,
            disable_persistent_padded_decode=False,
            persistent_padded_decode_steps=128,
            persistent_padded_full_attention_rows=False,
            persistent_padded_decode_cuda_graph=True,
            persistent_padded_decode_graph_warmup_iters=5,
            persistent_padded_sliding_metadata_padding=True,
            use_native_gemma_forward=True,
            native_gemma_attention_backend="sdpa",
            native_gemma_projection_backend="separate",
            native_gemma_weight_backend="hf_live",
            native_gemma_release_hf_decoder_layers=False,
            native_gemma_kv_sharing_fast_prefill=True,
            batched_routed_packets=True,
            routed_packet_workspace_bytes=32 * 1024 * 1024,
            enable_token_pool_metadata=True,
            enable_token_pool_attention=True,
            token_pool_max_context_len=16_384,
            token_pool_capacity=49_152,
            token_pool_paged_block_size=64,
            max_completed_requests=128,
        )

        kwargs = engine_kwargs_from_args(args)

        self.assertEqual(kwargs["prefill_chunk"], 2048)
        self.assertEqual(kwargs["prefill_microbatch_rows"], 1)
        self.assertEqual(kwargs["continuation_prefill_microbatch_rows"], 8)
        self.assertEqual(kwargs["decode_microbatch_rows"], 8)
        self.assertEqual(kwargs["decode_microbatch_bytes"], 370_000_000)
        self.assertEqual(kwargs["decode_batch_planner"], "length_bucketed")
        self.assertEqual(kwargs["decode_workspace_bytes"], 470_000_000)
        self.assertEqual(kwargs["decode_workspace_width_bucket"], 32)
        self.assertFalse(kwargs["persistent_exact_decode"])
        self.assertTrue(kwargs["persistent_padded_decode"])
        self.assertEqual(kwargs["persistent_padded_decode_steps"], 128)
        self.assertFalse(kwargs["persistent_padded_full_attention_rows"])
        self.assertTrue(kwargs["persistent_padded_decode_cuda_graph"])
        self.assertEqual(kwargs["persistent_padded_decode_graph_warmup_iters"], 5)
        self.assertTrue(kwargs["persistent_padded_sliding_metadata_padding"])
        self.assertTrue(kwargs["use_native_gemma_forward"])
        self.assertEqual(kwargs["native_gemma_attention_backend"], "sdpa")
        self.assertEqual(kwargs["native_gemma_projection_backend"], "separate")
        self.assertEqual(kwargs["native_gemma_weight_backend"], "hf_live")
        self.assertFalse(kwargs["native_gemma_release_hf_decoder_layers"])
        self.assertTrue(kwargs["native_gemma_kv_sharing_fast_prefill"])
        self.assertTrue(kwargs["batched_routed_packets"])
        self.assertEqual(kwargs["routed_packet_workspace_bytes"], 32 * 1024 * 1024)
        self.assertTrue(kwargs["enable_token_pool_metadata"])
        self.assertTrue(kwargs["enable_token_pool_attention"])
        self.assertEqual(kwargs["token_pool_max_context_len"], 16_384)
        self.assertEqual(kwargs["token_pool_capacity"], 49_152)
        self.assertEqual(kwargs["token_pool_paged_block_size"], 64)
        self.assertEqual(kwargs["finished_trace_limit"], 128)

    def test_scheduler_profile_uses_prefill_chunk_and_completion_lane(self) -> None:
        config = scheduler_config_from_args(
            SimpleNamespace(
                slots=4,
                prefill_chunk=2048,
                completion_prefill_lane_size=4,
            )
        )

        self.assertEqual(config.max_tokens_per_request_per_step, 2048)
        self.assertEqual(config.max_running_requests, 4)
        self.assertEqual(config.completion_prefill_lane_size, 4)
        self.assertEqual(config.max_tokens_per_step, 4 + 4 * 2048)

    def test_cli_rejects_invalid_prefill_profile_before_optional_imports(self) -> None:
        cases = (
            ("--prefill-chunk", "0", "--prefill-chunk must be >= 1"),
            (
                "--continuation-prefill-microbatch-rows",
                "-1",
                "--continuation-prefill-microbatch-rows must be >= 0 or omitted",
            ),
            (
                "--stream-flush-tokens",
                "0",
                "--stream-flush-tokens must be >= 1",
            ),
        )
        for option, value, expected in cases:
            with self.subTest(option=option):
                stderr = io.StringIO()
                with patch.object(sys, "stderr", stderr), patch.object(
                    sys,
                    "argv",
                    ["wkvm-gemma-server", "--model", "unused", option, value],
                ), self.assertRaises(SystemExit) as raised:
                    main()
                self.assertEqual(raised.exception.code, 2)
                self.assertIn(expected, stderr.getvalue())

    def test_chat_stop_tokens_are_opt_in_and_can_be_ignored(self) -> None:
        full_config = SimpleNamespace(eos_token_id=[1, 2])
        tokenizer = SimpleNamespace(eos_token_id=1, eot_token_id=106)

        self.assertEqual(
            _chat_stop_token_ids(full_config, None, ignore_eos=False),
            frozenset(),
        )
        self.assertEqual(
            _chat_stop_token_ids(full_config, tokenizer, ignore_eos=False),
            frozenset({1, 2, 106}),
        )
        self.assertEqual(
            _chat_stop_token_ids(full_config, tokenizer, ignore_eos=True),
            frozenset(),
        )

    def test_production_profile_uses_checkpoint_native_eager_decode_profile(self) -> None:
        args = SimpleNamespace(
            native_gemma_production_profile=True,
            native_gemma_checkpoint_loader=False,
            use_native_gemma_forward=False,
            native_gemma_attention_backend="manual",
            native_gemma_projection_backend="qkv_packed",
            native_gemma_weight_backend="owned",
            native_gemma_release_hf_decoder_layers=True,
            persistent_padded_decode_cuda_graph=False,
            persistent_padded_decode_steps=8,
            sink=8,
            window=512,
            m_slots=32,
            route_chunk=256,
            attn="sdpa",
        )

        apply_native_gemma_production_profile(args)
        validate_native_gemma_loader_args(args)

        self.assertTrue(args.native_gemma_checkpoint_loader)
        self.assertTrue(args.use_native_gemma_forward)
        self.assertEqual(args.native_gemma_attention_backend, "sdpa_single_gqa")
        self.assertEqual(args.native_gemma_projection_backend, "separate")
        self.assertEqual(args.native_gemma_weight_backend, "hf_live")
        self.assertFalse(args.native_gemma_release_hf_decoder_layers)
        self.assertFalse(args.persistent_padded_decode_cuda_graph)
        self.assertEqual(args.persistent_padded_decode_steps, 128)
        self.assertEqual(args.sink, 16)
        self.assertEqual(args.window, 1024)
        self.assertEqual(args.m_slots, 64)
        self.assertEqual(args.route_chunk, 512)
        self.assertEqual(args.attn, "eager")

    def test_checkpoint_native_loader_rejects_hf_release_modes(self) -> None:
        args = SimpleNamespace(
            native_gemma_checkpoint_loader=True,
            use_native_gemma_forward=False,
            native_gemma_weight_backend="owned",
            native_gemma_release_hf_decoder_layers=False,
        )

        with self.assertRaisesRegex(ValueError, "requires"):
            validate_native_gemma_loader_args(args)

        args.native_gemma_weight_backend = "hf_live"
        args.native_gemma_release_hf_decoder_layers = True
        with self.assertRaisesRegex(ValueError, "does not construct"):
            validate_native_gemma_loader_args(args)

        args.native_gemma_release_hf_decoder_layers = False
        validate_native_gemma_loader_args(args)
        self.assertTrue(args.use_native_gemma_forward)


class TestBoundedGemmaService(unittest.TestCase):
    @staticmethod
    def _decode_token_sse_writes(writes):
        events = []
        for block in b"".join(writes).decode().strip().split("\n\n"):
            data_line = next(
                line for line in block.splitlines() if line.startswith("data: ")
            )
            events.append(json.loads(data_line.removeprefix("data: ")))
        return events

    def test_token_sse_writer_batches_after_first_token_in_exact_order(self) -> None:
        wfile = RecordingWFile()
        writer = _TokenSSEWriter(wfile, flush_tokens=3)

        writer.send({"type": "queued", "req_id": "r"})
        self.assertEqual((len(wfile.writes), wfile.flush_calls), (1, 1))
        writer.send({"type": "token", "token": 1})
        self.assertEqual((len(wfile.writes), wfile.flush_calls), (2, 2))
        writer.send({"type": "token", "token": 2})
        writer.send({"type": "token", "token": 3})
        self.assertEqual((len(wfile.writes), wfile.flush_calls), (2, 2))
        writer.send({"type": "token", "token": 4})
        self.assertEqual((len(wfile.writes), wfile.flush_calls), (3, 3))
        writer.send({"type": "token", "token": 5})
        writer.send({"type": "finish", "finish_reason": "length"})
        self.assertEqual((len(wfile.writes), wfile.flush_calls), (4, 4))
        self.assertEqual(wfile.writes[2].count(b"event: token\n"), 3)
        self.assertIn(b"event: finish\n", wfile.writes[3])

        events = self._decode_token_sse_writes(wfile.writes)
        self.assertEqual(
            [event["type"] for event in events],
            ["queued", "token", "token", "token", "token", "token", "finish"],
        )
        self.assertEqual(
            [event["token"] for event in events if event["type"] == "token"],
            [1, 2, 3, 4, 5],
        )

    def test_token_sse_writer_default_preserves_per_event_writes(self) -> None:
        wfile = RecordingWFile()
        writer = _TokenSSEWriter(wfile)

        for event in (
            {"type": "queued"},
            {"type": "token", "token": 1},
            {"type": "token", "token": 2},
            {"type": "finish", "finish_reason": "length"},
        ):
            writer.send(event)

        self.assertEqual(len(wfile.writes), 4)
        self.assertEqual(wfile.flush_calls, 4)
        self.assertTrue(all(payload.count(b"\n\n") == 1 for payload in wfile.writes))

    def test_token_sse_disconnect_closes_source_iterator(self) -> None:
        wfile = RecordingWFile(fail_on_write=2)
        stream_iter = ClosableEventIterator(
            [
                {"type": "token", "token": 1},
                {"type": "finish", "finish_reason": "length"},
            ]
        )

        _write_token_sse_stream(
            wfile,
            stream_iter,
            {"type": "queued", "req_id": "r"},
            flush_tokens=8,
        )

        self.assertTrue(stream_iter.closed)
        self.assertEqual(len(wfile.writes), 1)
        self.assertEqual(wfile.flush_calls, 1)

    def test_http_listen_backlog_covers_b16_queue(self) -> None:
        service = BoundedGemmaService(
            FakeEngine(),
            max_queue=16,
            batch_wait_s=0.0,
        )
        server = serve(service, port=0)
        try:
            self.assertGreaterEqual(server.request_queue_size, 16)
        finally:
            service.close()
            server.server_close()

    def test_batch_wait_is_not_shortened_by_enqueue_notifications(self) -> None:
        engine = FakeEngine()
        original_step = engine.step
        step_times = []

        def recorded_step():
            step_times.append(time.perf_counter())
            return original_step()

        engine.step = recorded_step
        service = BoundedGemmaService(engine, max_queue=4, batch_wait_s=0.05)
        started = time.perf_counter()
        try:
            service.submit(prompt_ids=[1], max_tokens=1, req_id="first")
            time.sleep(0.01)
            service.submit(prompt_ids=[2], max_tokens=1, req_id="second")
            deadline = time.perf_counter() + 2
            while not step_times and time.perf_counter() < deadline:
                time.sleep(0.005)
            self.assertTrue(step_times)
            self.assertGreaterEqual(step_times[0] - started, 0.04)
        finally:
            service.close()

    def test_run_server_closes_service_and_restores_sigterm_handler(self) -> None:
        service = BoundedGemmaService(FakeEngine(), max_queue=2, batch_wait_s=0.0)
        server = SigtermServer()
        previous_sigterm_handler = signal.getsignal(signal.SIGTERM)

        run_server(service, server)

        self.assertTrue(service.closed)
        self.assertFalse(service._worker.is_alive())
        self.assertEqual(server.server_close_calls, 1)
        self.assertIs(signal.getsignal(signal.SIGTERM), previous_sigterm_handler)
        service.close()

    def test_close_fails_pending_requests_after_worker_stops(self) -> None:
        service = BoundedGemmaService(FakeEngine(), max_queue=2, batch_wait_s=60.0)
        service.submit(prompt_ids=[1], max_tokens=1, req_id="pending-close")

        service.close(timeout_s=1.0)

        self.assertFalse(service._worker.is_alive())
        self.assertEqual(len(service._pending), 0)
        status = service.status("pending-close")
        self.assertTrue(status["finished"])
        self.assertEqual(status["finish_reason"], "error")

    def test_close_reports_worker_that_does_not_stop_within_bound(self) -> None:
        engine = FinishWhileCancelWaitsEngine()
        service = BoundedGemmaService(engine, max_queue=2, batch_wait_s=0.0)
        service.submit(prompt_ids=[1], max_tokens=1, req_id="blocked-close")
        self.assertTrue(engine.step_started.wait(timeout=2))

        started = time.perf_counter()
        with self.assertRaisesRegex(RuntimeError, "did not stop within"):
            service.close(timeout_s=0.01)
        self.assertLess(time.perf_counter() - started, 0.5)
        self.assertTrue(service._worker.is_alive())

        engine.release_step.set()
        service.close(timeout_s=2.0)
        self.assertFalse(service._worker.is_alive())

    def test_generate_returns_tokens(self) -> None:
        service = BoundedGemmaService(FakeEngine(), max_queue=2)
        try:
            out = service.generate(prompt_ids=[1, 2], max_tokens=1, req_id="r")
            self.assertEqual(out["tokens"], [7])
            self.assertEqual(out["finish_reason"], "length")
        finally:
            service.close()

    def test_nonfinite_timeouts_are_rejected_before_admission(self) -> None:
        for value in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(default=value), self.assertRaisesRegex(
                ValueError,
                "finite",
            ):
                BoundedGemmaService(FakeEngine(), request_timeout_s=value)

        service = BoundedGemmaService(FakeEngine(), max_queue=2)
        try:
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(generate=value), self.assertRaisesRegex(
                    ValueError,
                    "finite",
                ):
                    service.generate(
                        prompt_ids=[1],
                        max_tokens=1,
                        req_id=f"generate-{value}",
                        timeout_s=value,
                    )
                with self.subTest(submit=value), self.assertRaisesRegex(
                    ValueError,
                    "finite",
                ):
                    service.submit(
                        prompt_ids=[1],
                        max_tokens=1,
                        req_id=f"submit-{value}",
                        timeout_s=value,
                    )
                with self.subTest(stream=value), self.assertRaisesRegex(
                    ValueError,
                    "finite",
                ):
                    next(
                        service.stream(
                            prompt_ids=[1],
                            max_tokens=1,
                            req_id=f"stream-{value}",
                            timeout_s=value,
                        )
                    )
            self.assertEqual(service.total_requests, 0)
        finally:
            service.close()

    def test_cli_rejects_nonfinite_request_timeout_before_optional_imports(self) -> None:
        stderr = io.StringIO()
        with patch.object(sys, "stderr", stderr), patch.object(
            sys,
            "argv",
            [
                "wkvm-gemma-server",
                "--model",
                "unused",
                "--request-timeout-s",
                "nan",
            ],
        ), self.assertRaises(SystemExit) as cm:
            main()

        self.assertEqual(cm.exception.code, 2)
        self.assertIn("--request-timeout-s must be finite and > 0", stderr.getvalue())

    def test_submit_status_returns_stepwise_tokens(self) -> None:
        service = BoundedGemmaService(FakeEngine(), max_queue=2, batch_wait_s=0.0)
        try:
            out = service.submit(prompt_ids=[1, 2], max_tokens=1, req_id="r")
            self.assertEqual(out["req_id"], "r")
            deadline = time.perf_counter() + 2
            status = service.status("r")
            while not status["finished"] and time.perf_counter() < deadline:
                time.sleep(0.01)
                status = service.status("r")
            self.assertTrue(status["finished"])
            self.assertEqual(status["tokens"], [7])
            self.assertEqual(status["finish_reason"], "length")
        finally:
            service.close()

    def test_queue_bound_rejects(self) -> None:
        service = BoundedGemmaService(FullEngine(), max_queue=1)
        try:
            with self.assertRaises(QueueFull):
                service.generate(prompt_ids=[1, 2], max_tokens=1)
        finally:
            service.close()

    def test_cancel_unknown_does_not_poison_future_request_id(self) -> None:
        service = BoundedGemmaService(FakeEngine(), max_queue=2, batch_wait_s=0.0)
        try:
            with self.assertRaisesRegex(KeyError, "unknown req_id future-request"):
                service.cancel("future-request")

            self.assertNotIn("future-request", service.cancelled)
            out = service.generate(
                prompt_ids=[1, 2],
                max_tokens=1,
                req_id="future-request",
            )
            self.assertEqual(out["tokens"], [7])
            self.assertEqual(out["finish_reason"], "length")
            self.assertNotIn("future-request", service.cancelled)
        finally:
            service.close()

    def test_cancelled_request_does_not_retain_tombstone(self) -> None:
        service = BoundedGemmaService(FakeEngine(), max_queue=2, batch_wait_s=0.05)
        try:
            service.submit(prompt_ids=[1, 2], max_tokens=1, req_id="cancel-me")
            self.assertEqual(
                service.cancel("cancel-me"),
                {"req_id": "cancel-me", "cancelled": True},
            )

            deadline = time.perf_counter() + 2
            status = service.status("cancel-me")
            while not status["finished"] and time.perf_counter() < deadline:
                time.sleep(0.01)
                status = service.status("cancel-me")
            self.assertTrue(status["finished"])
            self.assertEqual(status["finish_reason"], "aborted")
            self.assertNotIn("cancel-me", service.cancelled)
            self.assertEqual(service.total_cancelled, 1)
        finally:
            service.close()

    def test_cancel_rechecks_request_after_inflight_step(self) -> None:
        engine = FinishWhileCancelWaitsEngine()
        service = BoundedGemmaService(engine, max_queue=2, batch_wait_s=0.0)
        cancel_result = {}

        def cancel_request() -> None:
            cancel_result.update(service.cancel("race"))

        try:
            service.submit(prompt_ids=[1, 2], max_tokens=1, req_id="race")
            self.assertTrue(engine.step_started.wait(timeout=2))
            thread = threading.Thread(target=cancel_request)
            thread.start()
            deadline = time.perf_counter() + 2
            while "race" not in service.cancelled and time.perf_counter() < deadline:
                time.sleep(0.005)
            self.assertIn("race", service.cancelled)

            self.assertTrue(service.lock.acquire(timeout=0.1))
            service.lock.release()
            engine.release_step.set()
            thread.join(timeout=2)

            self.assertFalse(thread.is_alive())
            self.assertEqual(cancel_result, {"req_id": "race", "cancelled": False})
            self.assertEqual(service.status("race")["finish_reason"], "length")
            self.assertEqual(service.total_cancelled, 0)
            self.assertEqual(engine.abort_calls, 0)
            self.assertNotIn("race", service.cancelled)
        finally:
            engine.release_step.set()
            service.close()

    def test_http_cancel_unknown_returns_not_found(self) -> None:
        service = BoundedGemmaService(FakeEngine(), max_queue=2, batch_wait_s=0.0)
        server = serve(service, port=0)
        host, port = server.server_address
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            req = urllib.request.Request(
                f"http://{host}:{port}/v1/cancel",
                data=json.dumps({"req_id": "missing"}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(req, timeout=5)
            self.assertEqual(cm.exception.code, 404)
            with cm.exception as response:
                response.read()
            self.assertNotIn("missing", service.cancelled)
        finally:
            service.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_http_post_requires_content_length(self) -> None:
        service = BoundedGemmaService(FakeEngine(), max_queue=2, batch_wait_s=0.0)
        server = serve(service, port=0)
        host, port = server.server_address
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = http.client.HTTPConnection(host, port, timeout=2)
        try:
            connection.putrequest("POST", "/v1/submit")
            connection.putheader("Content-Type", "application/json")
            connection.endheaders()
            with connection.getresponse() as response:
                self.assertEqual(response.status, 411)
                payload = json.loads(response.read() or b"{}")
            self.assertIn("Content-Length", payload["error"])
            self.assertEqual(service.total_requests, 0)
        finally:
            connection.close()
            service.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_http_post_rejects_content_length_with_transfer_encoding(self) -> None:
        service = BoundedGemmaService(FakeEngine(), max_queue=2, batch_wait_s=0.0)
        server = serve(service, port=0)
        host, port = server.server_address
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = http.client.HTTPConnection(host, port, timeout=2)
        try:
            connection.putrequest("POST", "/v1/submit")
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Content-Length", "2")
            connection.putheader("Transfer-Encoding", "chunked")
            connection.endheaders()
            with connection.getresponse() as response:
                self.assertEqual(response.status, 400)
                payload = json.loads(response.read() or b"{}")
            self.assertIn("Transfer-Encoding", payload["error"])
            self.assertEqual(service.total_requests, 0)
        finally:
            connection.close()
            service.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_http_post_rejects_oversized_body_before_reading(self) -> None:
        body = json.dumps(
            {"prompt_ids": [1], "max_tokens": 1, "req_id": "bounded-body"},
            separators=(",", ":"),
        ).encode()
        service = BoundedGemmaService(FakeEngine(), max_queue=2, batch_wait_s=0.0)
        server = serve(service, port=0, max_request_body_bytes=len(body))
        host, port = server.server_address
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = http.client.HTTPConnection(host, port, timeout=2)
        try:
            connection.putrequest("POST", "/v1/submit")
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Content-Length", str(len(body) + 1))
            connection.endheaders()
            with connection.getresponse() as response:
                self.assertEqual(response.status, 413)
                payload = json.loads(response.read() or b"{}")
            self.assertIn("exceeds", payload["error"])
            self.assertEqual(service.total_requests, 0)

            request = urllib.request.Request(
                f"http://{host}:{port}/v1/submit",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=2) as response:
                self.assertEqual(response.status, 202)
                response.read()
            self.assertEqual(service.total_requests, 1)
        finally:
            connection.close()
            service.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_http_post_body_read_timeout_returns_request_timeout(self) -> None:
        body = b'{"prompt_ids":[1],"max_tokens":1}'
        service = BoundedGemmaService(FakeEngine(), max_queue=2, batch_wait_s=0.0)
        server = serve(service, port=0, request_read_timeout_s=0.05)
        host, port = server.server_address
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        connection = http.client.HTTPConnection(host, port, timeout=2)
        try:
            connection.putrequest("POST", "/v1/submit")
            connection.putheader("Content-Type", "application/json")
            connection.putheader("Content-Length", str(len(body)))
            connection.endheaders(body[:1])
            with connection.getresponse() as response:
                self.assertEqual(response.status, 408)
                payload = json.loads(response.read() or b"{}")
            self.assertIn("not received", payload["error"])
            self.assertEqual(service.total_requests, 0)
        finally:
            connection.close()
            service.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_http_submit_and_status_routes(self) -> None:
        service = BoundedGemmaService(FakeEngine(), max_queue=2, batch_wait_s=0.0)
        server = serve(service, port=0)
        host, port = server.server_address
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            body = json.dumps(
                {"prompt_ids": [1, 2], "max_tokens": 1, "req_id": "http-r"}
            ).encode()
            req = urllib.request.Request(
                f"http://{host}:{port}/v1/submit",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 202)
                submit = json.loads(resp.read() or b"{}")
            self.assertEqual(submit["req_id"], "http-r")

            deadline = time.perf_counter() + 2
            status = {}
            while time.perf_counter() < deadline:
                with urllib.request.urlopen(
                    f"http://{host}:{port}/v1/status/http-r", timeout=5
                ) as resp:
                    status = json.loads(resp.read() or b"{}")
                if status.get("finished"):
                    break
                time.sleep(0.01)
            self.assertTrue(status.get("finished"))
            self.assertEqual(status.get("tokens"), [7])
        finally:
            service.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_http_submit_accepts_timeout_s(self) -> None:
        service = BoundedGemmaService(FakeEngine(), max_queue=2, batch_wait_s=0.0)
        server = serve(service, port=0)
        host, port = server.server_address
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            body = json.dumps(
                {
                    "prompt_ids": [1, 2],
                    "max_tokens": 1,
                    "req_id": "http-timeout-ok",
                    "timeout_s": 5,
                }
            ).encode()
            req = urllib.request.Request(
                f"http://{host}:{port}/v1/submit",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 202)
                submit = json.loads(resp.read() or b"{}")
            self.assertEqual(submit["req_id"], "http-timeout-ok")
        finally:
            service.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_stream_yields_token_events_before_finish(self) -> None:
        service = BoundedGemmaService(StepwiseEngine(), max_queue=2, batch_wait_s=0.0)
        try:
            events = list(
                service.stream(
                    prompt_ids=[1, 2],
                    max_tokens=3,
                    req_id="stream-r",
                )
            )
            self.assertEqual([event["type"] for event in events], ["queued", "token", "token", "token", "finish"])
            self.assertEqual(
                [event["token"] for event in events if event["type"] == "token"],
                [100, 101, 102],
            )
            self.assertEqual(events[-1]["finish_reason"], "length")
        finally:
            service.close()

    def test_http_stream_route_returns_sse_events(self) -> None:
        service = BoundedGemmaService(StepwiseEngine(), max_queue=2, batch_wait_s=0.0)
        server = serve(service, port=0)
        host, port = server.server_address
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            body = json.dumps(
                {"prompt_ids": [1, 2], "max_tokens": 2, "req_id": "http-stream-r"}
            ).encode()
            req = urllib.request.Request(
                f"http://{host}:{port}/v1/stream",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                self.assertEqual(resp.headers.get_content_type(), "text/event-stream")
                body_text = resp.read().decode()
            events = []
            for block in body_text.strip().split("\n\n"):
                lines = block.splitlines()
                data_line = next(line for line in lines if line.startswith("data: "))
                events.append(json.loads(data_line.removeprefix("data: ")))
            self.assertEqual(
                [event["type"] for event in events],
                ["queued", "token", "token", "finish"],
            )
            self.assertEqual(
                [event["token"] for event in events if event["type"] == "token"],
                [100, 101],
            )
        finally:
            service.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_http_openai_completion_returns_blocking_response(self) -> None:
        service = BoundedGemmaService(StepwiseEngine(), max_queue=2, batch_wait_s=0.0)
        server = serve(service, port=0)
        host, port = server.server_address
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            body = json.dumps(
                {
                    "model": "gemma-test",
                    "prompt": [1, 2],
                    "max_tokens": 2,
                    "temperature": 0,
                    "stream": False,
                    "ignore_eos": True,
                    "return_token_ids": True,
                    "request_id": "cmpl-block",
                }
            ).encode()
            req = urllib.request.Request(
                f"http://{host}:{port}/v1/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                self.assertEqual(resp.headers.get_content_type(), "application/json")
                out = json.loads(resp.read() or b"{}")

            self.assertEqual(out["id"], "cmpl-block")
            self.assertEqual(out["object"], "text_completion")
            self.assertEqual(out["model"], "gemma-test")
            self.assertEqual(out["choices"][0]["finish_reason"], "length")
            self.assertEqual(out["choices"][0]["token_ids"], [100, 101])
            self.assertEqual(out["usage"]["prompt_tokens"], 2)
            self.assertEqual(out["usage"]["completion_tokens"], 2)
            self.assertEqual(out["usage"]["total_tokens"], 4)
        finally:
            service.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_http_openai_completion_streams_sse_chunks(self) -> None:
        service = BoundedGemmaService(StepwiseEngine(), max_queue=2, batch_wait_s=0.0)
        server = serve(service, port=0)
        host, port = server.server_address
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            body = json.dumps(
                {
                    "model": "gemma-test",
                    "prompt": [1, 2],
                    "max_tokens": 2,
                    "temperature": 0.0,
                    "stream": True,
                    "ignore_eos": True,
                    "return_token_ids": True,
                    "stream_options": {"include_usage": True},
                    "request_id": "cmpl-stream",
                }
            ).encode()
            req = urllib.request.Request(
                f"http://{host}:{port}/v1/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                self.assertEqual(resp.status, 200)
                self.assertEqual(resp.headers.get_content_type(), "text/event-stream")
                body_text = resp.read().decode()

            events = []
            for block in body_text.strip().split("\n\n"):
                data_line = next(line for line in block.splitlines() if line.startswith("data:"))
                data = data_line.removeprefix("data:").strip()
                events.append(data if data == "[DONE]" else json.loads(data))

            self.assertEqual(events[-1], "[DONE]")
            choice_events = [event for event in events if isinstance(event, dict) and event.get("choices")]
            self.assertEqual([event["choices"][0].get("token_ids") for event in choice_events[:2]], [[100], [101]])
            self.assertEqual(choice_events[-1]["choices"][0]["finish_reason"], "length")
            usage_events = [event for event in events if isinstance(event, dict) and event.get("usage")]
            self.assertEqual(len(usage_events), 1)
            self.assertEqual(usage_events[0]["choices"], [])
            self.assertEqual(usage_events[0]["usage"]["completion_tokens"], 2)
        finally:
            service.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_serving_benchmark_client_works_with_wkvm_openai_completion(self) -> None:
        service = BoundedGemmaService(StepwiseEngine(), max_queue=2, batch_wait_s=0.0)
        server = serve(service, port=0)
        host, port = server.server_address
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = stream_request_openai_completions(
                url=f"http://{host}:{port}",
                prompt=[1, 2],
                max_tokens=2,
                req_id="bench-client",
                timeout_s=5.0,
                model="gemma-test",
                extra_body=None,
            )

            self.assertTrue(result["success"])
            self.assertEqual(result["finish_reason"], "length")
            self.assertEqual(result["output_tokens"], 2)
            self.assertEqual(len(result["itl_s"]), 1)
            self.assertIsNone(result["error"])
        finally:
            service.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_http_openai_completion_rejects_unsupported_text_prompt(self) -> None:
        service = BoundedGemmaService(FakeEngine(), max_queue=2, batch_wait_s=0.0)
        server = serve(service, port=0)
        host, port = server.server_address
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            body = json.dumps(
                {"model": "gemma-test", "prompt": "hello", "max_tokens": 1}
            ).encode()
            req = urllib.request.Request(
                f"http://{host}:{port}/v1/completions",
                data=body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as cm:
                urllib.request.urlopen(req, timeout=5)
            self.assertEqual(cm.exception.code, 400)
            with cm.exception as resp:
                err = json.loads(resp.read() or b"{}")
            self.assertIn("token-id", err["error"])
        finally:
            service.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_generate_timeout_cancels_request(self) -> None:
        service = BoundedGemmaService(
            NeverFinishEngine(),
            max_queue=2,
            batch_wait_s=0.0,
            request_timeout_s=0.02,
        )
        try:
            with self.assertRaisesRegex(TimeoutError, "timed out"):
                service.generate(prompt_ids=[1, 2], max_tokens=1, req_id="slow")
            status = service.status("slow")
            self.assertTrue(status["finished"])
            self.assertEqual(status["finish_reason"], "aborted")
            self.assertEqual(service.metrics()["server"]["total_timed_out"], 1)
        finally:
            service.close()

    def test_generate_timeout_loses_race_to_completed_request(self) -> None:
        service = BoundedGemmaService(
            NeverFinishEngine(),
            max_queue=2,
            batch_wait_s=0.0,
        )
        original_timeout_request = service._timeout_request

        def finish_before_cancel(req_id: str) -> bool:
            with service.cv:
                request = service._requests[req_id]
                request.output_token_ids.append(7)
                request.status = type(request.status).FINISHED_LENGTH
            return original_timeout_request(req_id)

        service._timeout_request = finish_before_cancel
        try:
            out = service.generate(
                prompt_ids=[1, 2],
                max_tokens=1,
                req_id="timeout-race",
                timeout_s=0.001,
            )

            self.assertEqual(out["tokens"], [7])
            self.assertEqual(out["finish_reason"], "length")
            self.assertEqual(service.total_timed_out, 0)
        finally:
            service.close()

    def test_stream_timeout_loses_race_to_completed_request(self) -> None:
        service = BoundedGemmaService(
            NeverFinishEngine(),
            max_queue=2,
            batch_wait_s=0.0,
        )
        original_timeout_request = service._timeout_request

        def finish_before_cancel(req_id: str) -> bool:
            with service.cv:
                request = service._requests[req_id]
                request.output_token_ids.append(7)
                request.status = type(request.status).FINISHED_LENGTH
            return original_timeout_request(req_id)

        service._timeout_request = finish_before_cancel
        try:
            events = list(
                service.stream(
                    prompt_ids=[1, 2],
                    max_tokens=1,
                    req_id="stream-timeout-race",
                    timeout_s=0.001,
                )
            )

            self.assertEqual([event["type"] for event in events], ["queued", "token", "finish"])
            self.assertEqual(events[1]["token"], 7)
            self.assertEqual(events[2]["finish_reason"], "length")
            self.assertEqual(service.total_timed_out, 0)
        finally:
            service.close()

    def test_submit_timeout_expires_pending_request(self) -> None:
        service = BoundedGemmaService(
            FakeEngine(),
            max_queue=2,
            batch_wait_s=0.05,
        )
        try:
            service.submit(
                prompt_ids=[1, 2],
                max_tokens=1,
                req_id="pending-timeout",
                timeout_s=0.001,
            )
            deadline = time.perf_counter() + 2
            status = service.status("pending-timeout")
            while not status["finished"] and time.perf_counter() < deadline:
                time.sleep(0.01)
                status = service.status("pending-timeout")
            self.assertTrue(status["finished"])
            self.assertEqual(status["finish_reason"], "aborted")
            self.assertEqual(service.metrics()["server"]["total_timed_out"], 1)
        finally:
            service.close()

    def test_submit_deadline_removed_on_normal_completion(self) -> None:
        service = BoundedGemmaService(FakeEngine(), max_queue=2, batch_wait_s=0.0)
        try:
            service.submit(
                prompt_ids=[1, 2],
                max_tokens=1,
                req_id="deadline-done",
                timeout_s=5,
            )
            deadline = time.perf_counter() + 2
            status = service.status("deadline-done")
            while not status["finished"] and time.perf_counter() < deadline:
                time.sleep(0.01)
                status = service.status("deadline-done")
            self.assertTrue(status["finished"])
            self.assertEqual(status["finish_reason"], "length")
            self.assertNotIn("deadline-done", service._deadlines)
        finally:
            service.close()

    def test_worker_error_finishes_request_instead_of_hanging(self) -> None:
        service = BoundedGemmaService(ErrorEngine(), max_queue=2, batch_wait_s=0.0)
        try:
            out = service.generate(prompt_ids=[1, 2], max_tokens=1, req_id="err")
            self.assertEqual(out["finish_reason"], "error")
            self.assertIn("synthetic engine failure", service.last_error or "")
            self.assertEqual(service.total_errors, 1)
        finally:
            service.close()

    def test_fatal_worker_failure_fails_request_and_disables_admission(self) -> None:
        service = BoundedGemmaService(
            FatalAfterStepEngine(), max_queue=2, batch_wait_s=0.0
        )
        try:
            out = service.generate(prompt_ids=[1, 2], max_tokens=1, req_id="fatal")
            self.assertEqual(out["finish_reason"], "error")

            deadline = time.perf_counter() + 2
            while service._worker.is_alive() and time.perf_counter() < deadline:
                time.sleep(0.01)
            self.assertFalse(service._worker.is_alive())
            health = service.health()
            self.assertFalse(health["ok"])
            self.assertFalse(health["worker_alive"])
            self.assertIn("synthetic worker failure", health["last_error"] or "")
            with self.assertRaisesRegex(ServiceUnavailable, "service is not ready"):
                service.submit(prompt_ids=[1], max_tokens=1, req_id="rejected")
            self.assertNotIn("rejected", service._requests)
        finally:
            service.close()

    def test_fatal_worker_failure_releases_queued_waiter(self) -> None:
        service = BoundedGemmaService(FakeEngine(), max_queue=2, batch_wait_s=0.0)

        def fail_before_pending_drain() -> None:
            raise RuntimeError("synthetic pending-drain failure")

        service._expire_deadlines_locked = fail_before_pending_drain
        try:
            out = service.generate(prompt_ids=[1, 2], max_tokens=1, req_id="queued")
            self.assertEqual(out["finish_reason"], "error")
            self.assertEqual(len(service._pending), 0)

            deadline = time.perf_counter() + 2
            while service._worker.is_alive() and time.perf_counter() < deadline:
                time.sleep(0.01)
            health = service.health()
            self.assertFalse(health["ok"])
            self.assertEqual(health["pending_queue_depth"], 0)
            self.assertIn("pending-drain failure", health["last_error"] or "")
        finally:
            service.close()

    def test_http_health_and_admission_return_503_after_close(self) -> None:
        service = BoundedGemmaService(FakeEngine(), max_queue=2, batch_wait_s=0.0)
        server = serve(service, port=0)
        host, port = server.server_address
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            service.close()
            with self.assertRaises(urllib.error.HTTPError) as health_error:
                urllib.request.urlopen(f"http://{host}:{port}/health", timeout=5)
            self.assertEqual(health_error.exception.code, 503)
            with health_error.exception as response:
                health = json.loads(response.read() or b"{}")
            self.assertFalse(health["ok"])
            self.assertFalse(health["worker_alive"])

            request = urllib.request.Request(
                f"http://{host}:{port}/v1/submit",
                data=json.dumps({"prompt_ids": [1], "max_tokens": 1}).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with self.assertRaises(urllib.error.HTTPError) as submit_error:
                urllib.request.urlopen(request, timeout=5)
            self.assertEqual(submit_error.exception.code, 503)
            with submit_error.exception as response:
                response.read()
            self.assertEqual(service.total_requests, 0)
        finally:
            service.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_completed_request_retention_is_bounded(self) -> None:
        service = BoundedGemmaService(
            FakeEngine(),
            max_queue=2,
            batch_wait_s=0.0,
            max_completed_requests=2,
        )
        try:
            for i in range(3):
                out = service.generate(prompt_ids=[1, 2], max_tokens=1, req_id=f"r{i}")
                self.assertEqual(out["finish_reason"], "length")

            with self.assertRaises(KeyError):
                service.status("r0")
            self.assertEqual(service.status("r1")["finish_reason"], "length")
            self.assertEqual(service.status("r2")["finish_reason"], "length")
            metrics = service.metrics()["server"]
            self.assertEqual(metrics["tracked_requests"], 2)
            self.assertEqual(metrics["completed_tracked_requests"], 2)
        finally:
            service.close()


class TestTokenSessionStream(unittest.TestCase):
    def test_forced_stream_hides_candidate_before_overwrite_processing(self) -> None:
        turn = _ChatTurn(
            session_id="forced",
            response_id="turn",
            prompt_token_ids=[1],
            max_new_tokens=2,
            break_mask=None,
            deadline=None,
            session_kind="token",
            input_mode="initial_prompt",
            forced_output_token_ids=[91, 92],
            request=SimpleNamespace(output_token_ids=[65]),
            state="running",
        )

        self.assertEqual(BoundedGemmaService._chat_turn_tokens(turn), [])

    def test_forced_stream_uses_forced_prefix_before_request_overwrite(self) -> None:
        turn = _ChatTurn(
            session_id="forced",
            response_id="turn",
            prompt_token_ids=[1],
            max_new_tokens=2,
            break_mask=None,
            deadline=None,
            session_kind="token",
            input_mode="initial_prompt",
            forced_output_token_ids=[91, 92],
            request=SimpleNamespace(output_token_ids=[65, 13]),
            state="running",
            candidate_output_token_ids=[65],
        )

        self.assertEqual(BoundedGemmaService._chat_turn_tokens(turn), [91])

    @staticmethod
    def _start_server(engine, **service_kwargs):
        service = BoundedGemmaService(
            engine,
            max_queue=8,
            batch_wait_s=0.0,
            **service_kwargs,
        )
        server = serve(service, port=0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return service, server, thread

    @staticmethod
    def _stop_server(service, server, thread):
        service.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    @staticmethod
    def _post(host, port, body):
        request = urllib.request.Request(
            f"http://{host}:{port}/v1/stream",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.headers, response.read()

    @staticmethod
    def _sse_events(raw):
        events = []
        for block in raw.decode().strip().split("\n\n"):
            data_line = next(
                line for line in block.splitlines() if line.startswith("data: ")
            )
            events.append(json.loads(data_line.removeprefix("data: ")))
        return events

    def test_http_token_session_reuses_delta_continuation(self) -> None:
        engine = FakeSessionEngine(num_slots=2)
        service, server, thread = self._start_server(engine)
        host, port = server.server_address
        try:
            status, headers, raw = self._post(
                host,
                port,
                {
                    "session_id": "token-session",
                    "prompt_ids": [1, 2],
                    "max_tokens": 2,
                    "req_id": "token-turn-1",
                },
            )
            self.assertEqual(status, 200)
            self.assertEqual(headers.get_content_type(), "text/event-stream")
            first_events = self._sse_events(raw)
            self.assertEqual(
                [event["type"] for event in first_events],
                ["queued", "token", "token", "finish"],
            )
            self.assertEqual(
                [event["token"] for event in first_events if event["type"] == "token"],
                [65, 13],
            )
            first_metrics = first_events[-1]["metrics"]
            self.assertEqual(first_metrics["http_session_id"], "token-session")
            self.assertEqual(first_metrics["session_kind"], "token")
            self.assertEqual(first_metrics["session_input_mode"], "initial_prompt")
            self.assertEqual(first_metrics["session_input_tokens"], 2)
            self.assertFalse(first_metrics["session_reused"])

            _, _, raw = self._post(
                host,
                port,
                {
                    "session_id": "token-session",
                    "delta_ids": [9, 10],
                    "max_tokens": 2,
                    "req_id": "token-turn-2",
                },
            )
            second_events = self._sse_events(raw)
            second_metrics = second_events[-1]["metrics"]
            self.assertEqual(second_metrics["session_input_mode"], "continuation_delta")
            self.assertEqual(second_metrics["session_input_tokens"], 2)
            self.assertTrue(second_metrics["session_reused"])
            self.assertEqual(len(engine.started), 1)
            self.assertEqual(len(engine.continuations), 1)
            self.assertEqual(engine.continuations[0][1], [9, 10])
            self.assertEqual(service.health()["token_sessions"], 1)
        finally:
            self._stop_server(service, server, thread)

    def test_distinct_token_sessions_can_overlap(self) -> None:
        service = BoundedGemmaService(
            NeverFinishSessionEngine(num_slots=2),
            max_queue=4,
            batch_wait_s=0.0,
        )
        first = service.stream_token_session(
            session_id="first",
            prompt_ids=[1],
            max_tokens=2,
            req_id="first-turn",
        )
        second = service.stream_token_session(
            session_id="second",
            prompt_ids=[2],
            max_tokens=2,
            req_id="second-turn",
        )
        try:
            self.assertEqual(next(first)["type"], "queued")
            self.assertEqual(next(second)["type"], "queued")
            deadline = time.perf_counter() + 2
            while service.health()["token_sessions"] < 2 and time.perf_counter() < deadline:
                time.sleep(0.01)
            self.assertEqual(service.health()["token_sessions"], 2)
            self.assertEqual(set(service._chat_active_turns), {"first", "second"})
        finally:
            first.close()
            second.close()
            service.close()

    def test_duplicate_active_token_turn_returns_conflict(self) -> None:
        engine = NeverFinishSessionEngine(num_slots=1)
        service, server, thread = self._start_server(engine)
        host, port = server.server_address
        first = service.stream_token_session(
            session_id="busy",
            prompt_ids=[1],
            max_tokens=2,
            req_id="busy-turn-1",
        )
        try:
            self.assertEqual(next(first)["type"], "queued")
            with self.assertRaises(urllib.error.HTTPError) as raised:
                self._post(
                    host,
                    port,
                    {
                        "session_id": "busy",
                        "delta_ids": [2],
                        "max_tokens": 2,
                        "req_id": "busy-turn-2",
                    },
                )
            self.assertEqual(raised.exception.code, 409)
            payload = json.loads(raised.exception.read())
            self.assertEqual(payload["error"]["code"], "session_busy")
            self.assertIn("already has an active turn", payload["error"]["message"])
        finally:
            first.close()
            self._stop_server(service, server, thread)

    def test_http_token_session_invariants_are_explicit(self) -> None:
        service, server, thread = self._start_server(FakeSessionEngine(num_slots=2))
        host, port = server.server_address

        def assert_bad_request(body, message):
            with self.assertRaises(urllib.error.HTTPError) as raised:
                self._post(host, port, body)
            self.assertEqual(raised.exception.code, 400)
            payload = json.loads(raised.exception.read())
            self.assertIn(message, payload["error"])

        try:
            assert_bad_request(
                {"delta_ids": [1], "max_tokens": 1},
                "require an explicit session_id",
            )
            assert_bad_request(
                {
                    "session_id": "both",
                    "prompt_ids": [1],
                    "delta_ids": [2],
                    "max_tokens": 1,
                },
                "requires exactly one",
            )
            assert_bad_request(
                {
                    "session_id": "unknown",
                    "delta_ids": [1],
                    "max_tokens": 1,
                },
                "unknown token session unknown",
            )

            self._post(
                host,
                port,
                {
                    "session_id": "existing",
                    "prompt_ids": [1],
                    "max_tokens": 1,
                },
            )
            assert_bad_request(
                {
                    "session_id": "existing",
                    "prompt_ids": [2],
                    "max_tokens": 1,
                },
                "already exists; send delta_ids",
            )
        finally:
            self._stop_server(service, server, thread)

    def test_token_session_lru_and_ttl_close_parked_state(self) -> None:
        engine = FakeSessionEngine(num_slots=1)
        service = BoundedGemmaService(
            engine,
            max_queue=4,
            batch_wait_s=0.0,
            max_chat_sessions=1,
            chat_session_ttl_s=0.02,
        )
        try:
            list(
                service.stream_token_session(
                    session_id="first",
                    prompt_ids=[1],
                    max_tokens=1,
                    req_id="first-turn",
                )
            )
            first_engine_id = engine.started[0][0]
            deadline = time.perf_counter() + 2
            while (
                first_engine_id not in engine.closed_sessions
                and time.perf_counter() < deadline
            ):
                time.sleep(0.01)
            self.assertIn(first_engine_id, engine.closed_sessions)
            self.assertEqual(service.health()["token_sessions"], 0)

            service.chat_session_ttl_s = 60.0
            list(
                service.stream_token_session(
                    session_id="second",
                    prompt_ids=[2],
                    max_tokens=1,
                    req_id="second-turn",
                )
            )
            second_engine_id = engine.started[1][0]
            list(
                service.stream_token_session(
                    session_id="third",
                    prompt_ids=[3],
                    max_tokens=1,
                    req_id="third-turn",
                )
            )
            self.assertIn(second_engine_id, engine.closed_sessions)
            self.assertNotIn(second_engine_id, engine.scheduler.requests)
            self.assertEqual(service.health()["token_sessions"], 1)
        finally:
            service.close()

    def test_teacher_forcing_is_opt_in(self) -> None:
        service = BoundedGemmaService(
            FakeSessionEngine(num_slots=1),
            max_queue=2,
            batch_wait_s=0.0,
        )
        try:
            with self.assertRaisesRegex(ValueError, "teacher forcing is disabled"):
                service.stream_token_session(
                    session_id="disabled",
                    prompt_ids=[1],
                    forced_output_ids=[9],
                    max_tokens=1,
                    req_id="disabled-turn",
                )
        finally:
            service.close()

    def test_teacher_forced_outputs_and_continuation_reuse(self) -> None:
        engine = FakeSessionEngine(num_slots=1)
        service = BoundedGemmaService(
            engine,
            max_queue=4,
            batch_wait_s=0.0,
            enable_token_session_teacher_forcing=True,
        )
        try:
            first_events = list(
                service.stream_token_session(
                    session_id="forced",
                    prompt_ids=[1, 2],
                    forced_output_ids=[91, 92],
                    max_tokens=2,
                    req_id="forced-turn-1",
                )
            )
            self.assertEqual(
                [event["token"] for event in first_events if event["type"] == "token"],
                [91, 92],
            )
            forcing = first_events[-1]["metrics"]["teacher_forcing"]
            self.assertTrue(forcing["enabled"])
            self.assertEqual(forcing["candidate_output_ids"], [65, 13])
            self.assertEqual(forcing["forced_output_ids"], [91, 92])
            self.assertTrue(forcing["selected_outputs_match_forced"])
            self.assertFalse(forcing["candidate_outputs_match_forced"])
            self.assertEqual(forcing["scalar_overwrite_count"], 2)
            self.assertGreaterEqual(forcing["scalar_overwrite_s"], 0.0)
            self.assertEqual(
                forcing["overhead_contract"],
                {
                    "timed": True,
                    "full_vocabulary_mask": False,
                    "gpu_logit_elements_mutated_per_row": 0,
                    "mutation": "one_pending_token_scalar_overwrite",
                    "row_mutation_scope": "request_loop",
                },
            )

            second_events = list(
                service.stream_token_session(
                    session_id="forced",
                    delta_ids=[7],
                    forced_output_ids=[93, 94],
                    max_tokens=2,
                    req_id="forced-turn-2",
                )
            )
            second_finish = second_events[-1]
            self.assertEqual(
                [event["token"] for event in second_events if event["type"] == "token"],
                [93, 94],
            )
            self.assertTrue(second_finish["metrics"]["session_reused"])
            self.assertEqual(len(engine.started), 1)
            self.assertEqual(engine.continuations[0][1], [7])
            parked = next(iter(engine.scheduler.parked.values()))
            self.assertEqual(parked.prompt_token_ids, [1, 2, 91, 92, 7])
        finally:
            service.close()


class TestOpenAIChatCompatibility(unittest.TestCase):
    @staticmethod
    def _start_server(engine, **serve_kwargs):
        service = BoundedGemmaService(engine, max_queue=8, batch_wait_s=0.0)
        server = serve(service, port=0, **serve_kwargs)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return service, server, thread

    @staticmethod
    def _stop_server(service, server, thread):
        service.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    @staticmethod
    def _post(host, port, path, body, headers=None):
        request = urllib.request.Request(
            f"http://{host}:{port}{path}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", **(headers or {})},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, response.headers, response.read()

    @staticmethod
    def _sse_events(body):
        events = []
        for block in body.decode().strip().split("\n\n"):
            data_line = next(
                line for line in block.splitlines() if line.startswith("data:")
            )
            data = data_line.removeprefix("data:").strip()
            events.append(data if data == "[DONE]" else json.loads(data))
        return events

    def test_models_discovery_and_blocking_chat_response(self) -> None:
        service, server, thread = self._start_server(
            StepwiseEngine(),
            tokenizer=FakeChatTokenizer(),
            model_id="gemma-test",
            ignore_eos=True,
        )
        host, port = server.server_address
        try:
            with urllib.request.urlopen(
                f"http://{host}:{port}/v1/models", timeout=5
            ) as response:
                models = json.loads(response.read())
            self.assertEqual(models["object"], "list")
            self.assertEqual(models["data"][0]["id"], "gemma-test")
            self.assertEqual(models["data"][0]["owned_by"], "wkvm")

            status, headers, raw = self._post(
                host,
                port,
                "/v1/chat/completions",
                {
                    "model": "gemma-test",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "max_tokens": 2,
                    "temperature": 0,
                    "ignore_eos": True,
                    "stream": False,
                    "request_id": "chat-block",
                },
            )
            self.assertEqual(status, 200)
            self.assertEqual(headers.get_content_type(), "application/json")
            out = json.loads(raw)
            self.assertEqual(out["id"], "chat-block")
            self.assertEqual(out["object"], "chat.completion")
            self.assertEqual(out["choices"][0]["message"], {"role": "assistant", "content": "de"})
            self.assertEqual(out["choices"][0]["finish_reason"], "length")
            self.assertEqual(out["usage"]["completion_tokens"], 2)
        finally:
            self._stop_server(service, server, thread)

    def test_chat_rejects_per_request_eos_policy_mismatch(self) -> None:
        service, server, thread = self._start_server(
            StepwiseEngine(),
            tokenizer=FakeChatTokenizer(),
            model_id="gemma-test",
            ignore_eos=False,
        )
        host, port = server.server_address
        try:
            with self.assertRaises(urllib.error.HTTPError) as raised:
                self._post(
                    host,
                    port,
                    "/v1/chat/completions",
                    {
                        "model": "gemma-test",
                        "messages": [{"role": "user", "content": "Hi"}],
                        "max_tokens": 2,
                        "temperature": 0,
                        "ignore_eos": True,
                    },
                )
            self.assertEqual(raised.exception.code, 400)
            error = json.loads(raised.exception.read())
            self.assertIn("server EOS policy", error["error"])
        finally:
            self._stop_server(service, server, thread)

    def test_streaming_chat_reuses_openwebui_session_prefix(self) -> None:
        engine = FakeSessionEngine(num_slots=2)
        tokenizer = FakeChatTokenizer()
        service, server, thread = self._start_server(
            engine,
            tokenizer=tokenizer,
            model_id="gemma-test",
        )
        host, port = server.server_address
        try:
            first_messages = [{"role": "user", "content": "Hi"}]
            _, _, first_raw = self._post(
                host,
                port,
                "/v1/chat/completions",
                {
                    "model": "gemma-test",
                    "messages": first_messages,
                    "max_tokens": 2,
                    "temperature": 0,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "request_id": "chat-turn-1",
                },
                {"X-OpenWebUI-Chat-Id": "chat-123"},
            )
            first_events = self._sse_events(first_raw)
            first_text = "".join(
                event["choices"][0]["delta"].get("content", "")
                for event in first_events
                if isinstance(event, dict) and event.get("choices")
            )
            self.assertEqual(first_text, "A")
            self.assertEqual(first_events[-1], "[DONE]")

            second_messages = [
                *first_messages,
                {"role": "assistant", "content": "A"},
                {"role": "user", "content": "Next"},
            ]
            _, _, second_raw = self._post(
                host,
                port,
                "/v1/chat/completions",
                {
                    "model": "gemma-test",
                    "messages": second_messages,
                    "max_tokens": 2,
                    "temperature": 0,
                    "stream": True,
                    "request_id": "chat-turn-2",
                },
                {"X-OpenWebUI-Chat-Id": "chat-123"},
            )
            second_events = self._sse_events(second_raw)
            second_text = "".join(
                event["choices"][0]["delta"].get("content", "")
                for event in second_events
                if isinstance(event, dict) and event.get("choices")
            )
            self.assertEqual(second_text, "A")
            self.assertEqual(len(engine.started), 1)
            self.assertEqual(len(engine.continuations), 1)
            continuation = engine.continuations[0][1]
            first_prompt = tokenizer.apply_chat_template(
                first_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
            )
            second_prompt = tokenizer.apply_chat_template(
                second_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
            )
            self.assertEqual(continuation, second_prompt[len(first_prompt) + 2 :])
            self.assertFalse(engine.closed_sessions)
        finally:
            self._stop_server(service, server, thread)

    def test_openwebui_session_identity_isolated_by_user(self) -> None:
        engine = FakeSessionEngine(num_slots=2)
        service, server, thread = self._start_server(
            engine,
            tokenizer=FakeChatTokenizer(),
            model_id="gemma-test",
        )
        host, port = server.server_address
        try:
            first_messages = [{"role": "user", "content": "Hi"}]
            self._post(
                host,
                port,
                "/v1/chat/completions",
                {
                    "model": "gemma-test",
                    "messages": first_messages,
                    "max_tokens": 2,
                    "temperature": 0,
                    "request_id": "user-a-turn",
                },
                {
                    "X-OpenWebUI-User-Id": "user-a",
                    "X-OpenWebUI-Chat-Id": "shared-chat-id",
                },
            )
            self._post(
                host,
                port,
                "/v1/chat/completions",
                {
                    "model": "gemma-test",
                    "messages": [
                        *first_messages,
                        {"role": "assistant", "content": "A"},
                        {"role": "user", "content": "Next"},
                    ],
                    "max_tokens": 2,
                    "temperature": 0,
                    "request_id": "user-b-turn",
                },
                {
                    "X-OpenWebUI-User-Id": "user-b",
                    "X-OpenWebUI-Chat-Id": "shared-chat-id",
                },
            )

            self.assertEqual(len(engine.started), 2)
            self.assertFalse(engine.continuations)
            self.assertFalse(engine.closed_sessions)
            self.assertEqual(service.metrics()["server"]["chat_sessions"], 2)
        finally:
            self._stop_server(service, server, thread)

    def test_prefix_mismatch_restarts_and_retires_old_session(self) -> None:
        engine = FakeSessionEngine(num_slots=1)
        service = BoundedGemmaService(
            engine,
            max_queue=4,
            batch_wait_s=0.0,
            max_chat_sessions=1,
        )
        try:
            list(
                service.stream_chat(
                    prompt_ids=[1, 2],
                    max_tokens=2,
                    session_id="session",
                    req_id="turn-1",
                    break_mask=[False, False],
                )
            )
            old_engine_id = engine.started[0][0]
            list(
                service.stream_chat(
                    prompt_ids=[1, 3, 4],
                    max_tokens=2,
                    session_id="session",
                    req_id="turn-2",
                    break_mask=[False, False, False],
                )
            )
            self.assertEqual(len(engine.started), 2)
            self.assertFalse(engine.continuations)
            self.assertEqual(engine.closed_sessions, [old_engine_id])
            self.assertNotIn(old_engine_id, engine.scheduler.requests)
        finally:
            service.close()

    def test_chat_cohort_empties_cuda_cache_once_per_turn(self) -> None:
        empty_cache_calls = []
        engine = FakeSessionEngine(num_slots=1)
        service = BoundedGemmaService(
            engine,
            max_queue=4,
            batch_wait_s=0.0,
            cuda_empty_cache=lambda: empty_cache_calls.append(time.perf_counter()),
        )
        try:
            list(
                service.stream_chat(
                    prompt_ids=[1],
                    max_tokens=2,
                    session_id="session",
                    req_id="turn-1",
                )
            )
            list(
                service.stream_chat(
                    prompt_ids=[1, 65, 13, 2],
                    max_tokens=2,
                    session_id="session",
                    req_id="turn-2",
                )
            )
            self.assertEqual(len(empty_cache_calls), 2)
            server_metrics = service.metrics()["server"]
            self.assertTrue(server_metrics["empty_cuda_cache_before_decode"])
            self.assertEqual(server_metrics["cuda_empty_cache_calls"], 2)
        finally:
            service.close()

    def test_lru_and_ttl_retire_parked_session_history(self) -> None:
        engine = FakeSessionEngine(num_slots=1)
        service = BoundedGemmaService(
            engine,
            max_queue=4,
            batch_wait_s=0.0,
            max_chat_sessions=1,
            chat_session_ttl_s=0.02,
        )
        try:
            list(
                service.stream_chat(
                    prompt_ids=[1],
                    max_tokens=1,
                    session_id="first",
                    req_id="first-turn",
                )
            )
            first_engine_id = engine.started[0][0]
            deadline = time.perf_counter() + 2
            while first_engine_id not in engine.closed_sessions and time.perf_counter() < deadline:
                time.sleep(0.01)
            self.assertIn(first_engine_id, engine.closed_sessions)
            self.assertNotIn(first_engine_id, engine.scheduler.requests)
            self.assertEqual(service.metrics()["server"]["chat_sessions"], 0)
            service.chat_session_ttl_s = 60.0

            list(
                service.stream_chat(
                    prompt_ids=[2],
                    max_tokens=1,
                    session_id="second",
                    req_id="second-turn",
                )
            )
            list(
                service.stream_chat(
                    prompt_ids=[3],
                    max_tokens=1,
                    session_id="third",
                    req_id="third-turn",
                )
            )
            second_engine_id = engine.started[1][0]
            self.assertIn(second_engine_id, engine.closed_sessions)
            self.assertNotIn(second_engine_id, engine.scheduler.requests)
        finally:
            service.close()

    def test_same_session_rejects_overlapping_turn(self) -> None:
        service = BoundedGemmaService(
            NeverFinishSessionEngine(num_slots=1),
            max_queue=4,
            batch_wait_s=0.0,
        )
        first = service.stream_chat(
            prompt_ids=[1],
            max_tokens=2,
            session_id="busy",
            req_id="turn-1",
        )
        try:
            self.assertEqual(next(first)["type"], "queued")
            second = service.stream_chat(
                prompt_ids=[1, 2],
                max_tokens=2,
                session_id="busy",
                req_id="turn-2",
            )
            with self.assertRaisesRegex(RuntimeError, "active turn"):
                next(second)
        finally:
            first.close()
            service.close()


if __name__ == "__main__":
    unittest.main()
