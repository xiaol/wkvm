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
from types import SimpleNamespace
from unittest.mock import patch

from experiments.wkvm_serving_bench import stream_request_openai_completions
from wkvm.gemma_server import (
    BoundedGemmaService,
    QueueFull,
    ServiceUnavailable,
    apply_native_gemma_production_profile,
    engine_kwargs_from_args,
    main,
    run_server,
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


class TestGemmaServerEngineArgs(unittest.TestCase):
    def test_native_forward_flags_are_passed_to_engine_kwargs(self) -> None:
        args = SimpleNamespace(
            decode_microbatch_rows=8,
            decode_microbatch_bytes=370_000_000,
            decode_batch_planner="length_bucketed",
            decode_workspace_bytes=470_000_000,
            decode_workspace_width_bucket=32,
            disable_persistent_exact_decode=True,
            disable_persistent_padded_decode=False,
            persistent_padded_decode_steps=128,
            persistent_padded_decode_cuda_graph=True,
            persistent_padded_decode_graph_warmup_iters=5,
            persistent_padded_sliding_metadata_padding=True,
            use_native_gemma_forward=True,
            native_gemma_attention_backend="sdpa",
            native_gemma_projection_backend="separate",
            native_gemma_weight_backend="hf_live",
            native_gemma_release_hf_decoder_layers=False,
            enable_token_pool_metadata=True,
            enable_token_pool_attention=True,
            token_pool_max_context_len=16_384,
            token_pool_capacity=49_152,
            token_pool_paged_block_size=64,
            max_completed_requests=128,
        )

        kwargs = engine_kwargs_from_args(args)

        self.assertEqual(kwargs["decode_microbatch_rows"], 8)
        self.assertEqual(kwargs["decode_microbatch_bytes"], 370_000_000)
        self.assertEqual(kwargs["decode_batch_planner"], "length_bucketed")
        self.assertEqual(kwargs["decode_workspace_bytes"], 470_000_000)
        self.assertEqual(kwargs["decode_workspace_width_bucket"], 32)
        self.assertFalse(kwargs["persistent_exact_decode"])
        self.assertTrue(kwargs["persistent_padded_decode"])
        self.assertEqual(kwargs["persistent_padded_decode_steps"], 128)
        self.assertTrue(kwargs["persistent_padded_decode_cuda_graph"])
        self.assertEqual(kwargs["persistent_padded_decode_graph_warmup_iters"], 5)
        self.assertTrue(kwargs["persistent_padded_sliding_metadata_padding"])
        self.assertTrue(kwargs["use_native_gemma_forward"])
        self.assertEqual(kwargs["native_gemma_attention_backend"], "sdpa")
        self.assertEqual(kwargs["native_gemma_projection_backend"], "separate")
        self.assertEqual(kwargs["native_gemma_weight_backend"], "hf_live")
        self.assertFalse(kwargs["native_gemma_release_hf_decoder_layers"])
        self.assertTrue(kwargs["enable_token_pool_metadata"])
        self.assertTrue(kwargs["enable_token_pool_attention"])
        self.assertEqual(kwargs["token_pool_max_context_len"], 16_384)
        self.assertEqual(kwargs["token_pool_capacity"], 49_152)
        self.assertEqual(kwargs["token_pool_paged_block_size"], 64)
        self.assertEqual(kwargs["finished_trace_limit"], 128)

    def test_production_profile_enables_checkpoint_native_graph_profile(self) -> None:
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
        self.assertTrue(args.persistent_padded_decode_cuda_graph)
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


if __name__ == "__main__":
    unittest.main()
