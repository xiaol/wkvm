import json
import types
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import experiments.wkvm_serving_bench as bench
from experiments.wkvm_serving_bench import (
    openai_delta_token_count,
    build_prompts,
    parse_sse_line,
    run_warmup,
    sse_events_from_line,
    stream_request_openai_completions,
    summarize_row,
)


class _CompletionHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002
        return

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        self.server.seen_path = self.path  # type: ignore[attr-defined]
        self.server.seen_body = json.loads(body)  # type: ignore[attr-defined]
        if self.path != "/v1/completions":
            self.send_response(404)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        for event in self.server.events:  # type: ignore[attr-defined]
            if event == "[DONE]":
                chunk = "data: [DONE]\n\n"
            else:
                chunk = "data: " + json.dumps(event) + "\n\n"
            self.wfile.write(chunk.encode())
            self.wfile.flush()
        self.close_connection = True


class _FakeOpenAIServer:
    def __init__(self, events):
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _CompletionHandler)
        self._server.events = events
        self._server.seen_path = None
        self._server.seen_body = None
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        host, port = self._server.server_address
        return self, f"http://{host}:{port}"

    def __exit__(self, exc_type, exc, tb):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2.0)

    @property
    def seen_body(self):
        return self._server.seen_body


class TestWkvmServingBench(unittest.TestCase):
    def test_parse_sse_line_accepts_done_with_or_without_space(self) -> None:
        self.assertEqual(parse_sse_line("data: [DONE]"), "[DONE]")
        self.assertEqual(parse_sse_line("data:[DONE]"), "[DONE]")
        self.assertIsNone(parse_sse_line(": keepalive"))

    def test_sse_events_from_line_parses_multiple_data_lines(self) -> None:
        event = {"choices": [{"token_ids": [101], "finish_reason": None}]}
        raw = (
            b": ping\n\n"
            + ("data: " + json.dumps(event) + "\n\n").encode()
            + b"data:[DONE]\n\n"
        )

        self.assertEqual(sse_events_from_line(raw), [event, "[DONE]"])

    def test_openai_delta_token_count_handles_common_stream_shapes(self) -> None:
        self.assertEqual(openai_delta_token_count({"token_ids": [1, 2]}), 2)
        self.assertEqual(openai_delta_token_count({"logprobs": {"tokens": ["a", "b"]}}), 2)
        self.assertEqual(openai_delta_token_count({"text": "hello", "finish_reason": None}), 1)
        self.assertEqual(openai_delta_token_count({"text": "", "finish_reason": "length"}), 0)

    def test_stream_request_openai_completions_posts_token_id_prompt(self) -> None:
        events = [
            {"choices": [{"token_ids": [101], "finish_reason": None}]},
            {"choices": [{"token_ids": [102], "finish_reason": None}]},
            {
                "choices": [{"text": "", "finish_reason": "length"}],
                "usage": {"completion_tokens": 2},
            },
            "[DONE]",
        ]
        with _FakeOpenAIServer(events) as (server, url):
            result = stream_request_openai_completions(
                url=url,
                prompt=[1, 2, 3],
                max_tokens=2,
                req_id="openai-1",
                timeout_s=5.0,
                model="gemma-test",
                extra_body={"ignore_eos": False, "top_p": 1.0},
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["output_tokens"], 2)
        self.assertEqual(result["finish_reason"], "length")
        self.assertEqual(len(result["itl_s"]), 1)
        self.assertIsNotNone(result["ttft_s"])
        self.assertEqual(server.seen_body["model"], "gemma-test")
        self.assertEqual(server.seen_body["prompt"], [1, 2, 3])
        self.assertEqual(server.seen_body["max_tokens"], 2)
        self.assertTrue(server.seen_body["stream"])
        self.assertTrue(server.seen_body["return_token_ids"])
        self.assertFalse(server.seen_body["ignore_eos"])
        self.assertEqual(server.seen_body["top_p"], 1.0)
        self.assertEqual(server.seen_body["stream_options"], {"include_usage": True})

    def test_summarize_row_reports_streaming_latency_and_throughput(self) -> None:
        row = summarize_row(
            2,
            [
                {
                    "req_id": "a",
                    "success": True,
                    "finish_reason": "length",
                    "error": None,
                    "output_tokens": 3,
                    "ttft_s": 0.10,
                    "e2e_latency_s": 0.40,
                    "itl_s": [0.10, 0.20],
                },
                {
                    "req_id": "b",
                    "success": True,
                    "finish_reason": "length",
                    "error": None,
                    "output_tokens": 3,
                    "ttft_s": 0.20,
                    "e2e_latency_s": 0.60,
                    "itl_s": [0.20, 0.30],
                },
            ],
            elapsed_s=0.75,
        )

        self.assertEqual(row["success_count"], 2)
        self.assertEqual(row["error_count"], 0)
        self.assertEqual(row["output_tokens"], 6)
        self.assertEqual(row["request_output_tok_s"], 8.0)
        self.assertEqual(row["p50_ttft_s"], 0.15)
        self.assertEqual(row["p95_e2e_latency_s"], 0.59)
        self.assertEqual(row["p50_itl_s"], 0.20)

    def test_summarize_row_carries_error_samples(self) -> None:
        row = summarize_row(
            1,
            [
                {
                    "req_id": "bad",
                    "success": False,
                    "finish_reason": "error",
                    "error": "boom",
                    "output_tokens": 0,
                    "ttft_s": None,
                    "e2e_latency_s": 0.01,
                    "itl_s": [],
                }
            ],
            elapsed_s=0.01,
        )

        self.assertEqual(row["success_count"], 0)
        self.assertEqual(row["error_count"], 1)
        self.assertEqual(row["request_output_tok_s"], 0.0)
        self.assertEqual(row["errors"], [{"req_id": "bad", "error": "boom", "finish_reason": "error"}])

    def test_run_warmup_uses_same_request_path_and_caps_to_row_concurrency(self) -> None:
        calls = []

        def fake_stream_request(**kwargs):
            calls.append(kwargs)
            return {
                "req_id": kwargs["req_id"],
                "success": True,
                "finish_reason": "length",
                "error": None,
                "output_tokens": kwargs["max_tokens"],
                "ttft_s": 0.01,
                "e2e_latency_s": 0.02,
                "itl_s": [],
            }

        args = types.SimpleNamespace(
            backend="openai-completions",
            engine="test-engine",
            ctx=8,
            warmup_requests=4,
            warmup_output_tokens=2,
            request_timeout_s=5.0,
            served_model="gemma-test",
        )
        old_stream_request = bench.stream_request
        try:
            bench.stream_request = fake_stream_request
            summary = run_warmup(
                "http://127.0.0.1:1",
                2,
                [[1, 2], [3, 4]],
                args,
                extra_body={"ignore_eos": True},
            )
        finally:
            bench.stream_request = old_stream_request

        self.assertEqual(len(calls), 2)
        self.assertEqual(summary["success_count"], 2)
        self.assertEqual(summary["output_tokens"], 4)
        self.assertEqual(summary["requested_output_tokens"], 2)
        self.assertEqual(summary["prompt_lengths"], [2, 2])
        self.assertTrue(all(call["max_tokens"] == 2 for call in calls))
        self.assertEqual([call["req_id"] for call in calls], ["warmup-2-0", "warmup-2-1"])

    def test_run_warmup_returns_none_when_disabled(self) -> None:
        args = types.SimpleNamespace(warmup_requests=0)
        self.assertIsNone(
            run_warmup(
                "http://127.0.0.1:1",
                1,
                [[1]],
                args,
                extra_body=None,
            )
        )

    def test_build_prompts_row_offset_changes_content_not_length(self) -> None:
        args = types.SimpleNamespace(
            model_path="/run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it",
            concurrency=[1, 2],
            ctx=96,
            prompt_lengths="uniform",
        )
        try:
            measured = build_prompts(args, row_offset=0)
            warmup = build_prompts(args, row_offset=64)
        except Exception as exc:
            self.skipTest(f"local tokenizer unavailable: {exc}")

        for B in args.concurrency:
            self.assertEqual([len(p) for p in measured[B]], [len(p) for p in warmup[B]])
            self.assertNotEqual(measured[B][0], warmup[B][0])


if __name__ == "__main__":
    unittest.main()
