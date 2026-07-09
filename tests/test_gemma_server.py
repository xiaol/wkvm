import unittest
import threading
import time
import urllib.error
import urllib.request
import json
from types import SimpleNamespace

from experiments.wkvm_serving_bench import stream_request_openai_completions
from wkvm.gemma_server import (
    BoundedGemmaService,
    QueueFull,
    apply_native_gemma_production_profile,
    engine_kwargs_from_args,
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


class FullQueue(FakeQueue):
    def __len__(self):
        return 1


class FullEngine(FakeEngine):
    def __init__(self):
        super().__init__()
        self.scheduler.waiting = FullQueue()


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
            use_native_gemma_forward=True,
            native_gemma_attention_backend="sdpa",
            native_gemma_projection_backend="separate",
            native_gemma_weight_backend="hf_live",
            native_gemma_release_hf_decoder_layers=False,
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
        self.assertTrue(kwargs["use_native_gemma_forward"])
        self.assertEqual(kwargs["native_gemma_attention_backend"], "sdpa")
        self.assertEqual(kwargs["native_gemma_projection_backend"], "separate")
        self.assertEqual(kwargs["native_gemma_weight_backend"], "hf_live")
        self.assertFalse(kwargs["native_gemma_release_hf_decoder_layers"])
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
    def test_generate_returns_tokens(self) -> None:
        service = BoundedGemmaService(FakeEngine(), max_queue=2)
        try:
            out = service.generate(prompt_ids=[1, 2], max_tokens=1, req_id="r")
            self.assertEqual(out["tokens"], [7])
            self.assertEqual(out["finish_reason"], "length")
        finally:
            service.close()

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
