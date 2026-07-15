import json
import types
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import experiments.wkvm_serving_bench as bench
from experiments.wkvm_serving_bench import (
    WholeGpuMemoryMonitor,
    benchmark_request_id,
    build_provenance,
    build_prompts,
    openai_delta_token_count,
    openai_delta_token_info,
    parse_sse_line,
    parse_json_object,
    run_row,
    run_warmup,
    sse_events_from_line,
    stream_request_openai_completions,
    summarize_row,
    validate_extra_body,
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
    def test_query_nvidia_gpu_returns_structured_identity_and_memory(self) -> None:
        calls = []

        def fake_check_output(command, **kwargs):
            calls.append((command, kwargs))
            return (
                "0, GPU-test, NVIDIA Test GPU, 595.71.05, 24564, 12345\n"
            )

        old_check_output = bench.subprocess.check_output
        try:
            bench.subprocess.check_output = fake_check_output
            gpu = bench.query_nvidia_gpu("GPU-test")
        finally:
            bench.subprocess.check_output = old_check_output

        self.assertEqual(gpu["index"], 0)
        self.assertEqual(gpu["uuid"], "GPU-test")
        self.assertEqual(gpu["driver_version"], "595.71.05")
        self.assertEqual(gpu["memory_total_mib"], 24564)
        self.assertEqual(gpu["memory_used_mib"], 12345)
        self.assertIn("--id=GPU-test", calls[0][0])

    def test_whole_gpu_monitor_records_baseline_peak_and_scope(self) -> None:
        samples = iter(
            [
                {
                    "index": 0,
                    "uuid": "GPU-test",
                    "name": "Test GPU",
                    "driver_version": "1.2.3",
                    "memory_total_mib": 24000,
                    "memory_used_mib": 1000,
                },
                {
                    "index": 0,
                    "uuid": "GPU-test",
                    "name": "Test GPU",
                    "driver_version": "1.2.3",
                    "memory_total_mib": 24000,
                    "memory_used_mib": 1128,
                },
            ]
        )
        old_query = bench.query_nvidia_gpu
        try:
            bench.query_nvidia_gpu = lambda _device: next(samples)
            with WholeGpuMemoryMonitor("0", interval_s=60.0) as monitor:
                pass
        finally:
            bench.query_nvidia_gpu = old_query

        result = monitor.result()
        self.assertEqual(result["schema"], "wkvm.whole_gpu_memory.v1")
        self.assertEqual(result["scope"], "whole_device")
        self.assertEqual(result["baseline_used_mib"], 1000)
        self.assertEqual(result["peak_used_mib"], 1128)
        self.assertEqual(result["peak_delta_mib"], 128)
        self.assertEqual(result["sample_count"], 2)
        self.assertEqual(result["device_uuid"], "GPU-test")
        self.assertEqual(result["gpu_name"], "Test GPU")
        self.assertEqual(result["driver_version"], "1.2.3")
        self.assertEqual(result["memory_total_mib"], 24000)

    def test_provenance_separates_server_version_from_client_packages(self) -> None:
        args = types.SimpleNamespace(
            engine="vllm-http-stream",
            engine_version="0.24.0",
            engine_version_source="server_environment",
            target_server_launch_command="vllm serve /models/gemma --port 8001",
            target_server_config={"tensor_parallel_size": 1},
            gpu_memory_device="0",
            gpu_memory_sample_interval_s=0.2,
        )
        old_versions = bench.installed_package_versions
        try:
            bench.installed_package_versions = lambda: {
                "wkvm": "0.0.1",
                "torch": "2.10.0",
                "transformers": None,
                "vllm": "0.23.0",
                "sglang": None,
            }
            provenance = build_provenance(
                args,
                commit="a" * 40,
                gpu={
                    "index": 0,
                    "uuid": "GPU-test",
                    "name": "Test GPU",
                    "driver_version": "1.2.3",
                    "memory_total_mib": 24000,
                    "source": "nvidia-smi",
                },
                gpu_probe_error=None,
            )
        finally:
            bench.installed_package_versions = old_versions

        self.assertEqual(provenance["schema"], "wkvm.serving_bench.provenance.v2")
        self.assertEqual(provenance["engine"]["version"], "0.24.0")
        self.assertEqual(
            provenance["engine"]["version_source"], "server_environment"
        )
        self.assertEqual(
            provenance["client_environment"]["packages"]["vllm"], "0.23.0"
        )
        self.assertEqual(provenance["gpu"]["driver_version"], "1.2.3")
        self.assertEqual(
            provenance["target_server"]["launch_command"],
            "vllm serve /models/gemma --port 8001",
        )
        self.assertEqual(
            provenance["target_server"]["launch_command_source"],
            "operator_supplied",
        )
        self.assertEqual(
            provenance["target_server"]["config"], {"tensor_parallel_size": 1}
        )
        self.assertTrue(provenance["gpu_memory_monitor"]["enabled"])
        self.assertIn("every process", provenance["gpu_memory_monitor"]["caveat"])

    def test_target_server_config_parser_requires_a_json_object(self) -> None:
        self.assertEqual(
            parse_json_object('{"tensor_parallel_size": 1}'),
            {"tensor_parallel_size": 1},
        )
        with self.assertRaises(bench.argparse.ArgumentTypeError):
            parse_json_object("[]")
        with self.assertRaises(bench.argparse.ArgumentTypeError):
            parse_json_object("not-json")

    def test_requested_gpu_monitor_fails_before_prompt_construction(self) -> None:
        args = types.SimpleNamespace(
            run_id="run-test",
            extra_body_json=None,
            url="http://127.0.0.1:1",
            gpu_memory_device="missing",
        )
        old_collect = bench.collect_gpu_provenance
        old_build_prompts = bench.build_prompts
        try:
            bench.collect_gpu_provenance = lambda _device: (None, "device not found")
            bench.build_prompts = lambda _args: self.fail(
                "prompt construction must not run after a failed GPU probe"
            )
            with self.assertRaisesRegex(RuntimeError, "device not found"):
                bench.run(args)
        finally:
            bench.collect_gpu_provenance = old_collect
            bench.build_prompts = old_build_prompts

    def test_request_ids_are_namespaced_per_run(self) -> None:
        args = types.SimpleNamespace(run_id="run-123")
        self.assertEqual(
            benchmark_request_id(args, "serve", 8, 2),
            "run-123-serve-8-2",
        )

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
        self.assertEqual(
            openai_delta_token_info({"token_ids": [1]}),
            (1, "token_ids", True),
        )
        self.assertEqual(
            openai_delta_token_info({"text": "multiple tokens may be here"}),
            (1, "text_chunk", False),
        )

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
                extra_body={"priority": 0},
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
        self.assertTrue(server.seen_body["ignore_eos"])
        self.assertEqual(server.seen_body["priority"], 0)
        self.assertEqual(server.seen_body["stream_options"], {"include_usage": True})
        self.assertTrue(result["itl_valid"])
        self.assertTrue(result["output_token_count_exact"])
        self.assertEqual(result["output_token_count_source"], "usage")

    def test_openai_text_chunks_do_not_claim_token_exact_itl(self) -> None:
        events = [
            {"choices": [{"text": "two tokens", "finish_reason": None}]},
            {
                "choices": [{"text": "", "finish_reason": "length"}],
                "usage": {"completion_tokens": 2},
            },
            "[DONE]",
        ]
        with _FakeOpenAIServer(events) as (_server, url):
            result = stream_request_openai_completions(
                url=url,
                prompt=[1, 2, 3],
                max_tokens=2,
                req_id="openai-text",
                timeout_s=5.0,
                model="gemma-test",
            )

        self.assertTrue(result["success"])
        self.assertFalse(result["itl_valid"])
        self.assertTrue(result["output_token_count_exact"])
        self.assertEqual(result["output_token_count_source"], "usage")
        self.assertEqual(result["stream_token_count_sources"], ["text_chunk"])

    def test_openai_exact_stream_rejects_usage_count_disagreement(self) -> None:
        events = [
            {"choices": [{"token_ids": [101], "finish_reason": None}]},
            {"choices": [{"token_ids": [102], "finish_reason": None}]},
            {
                "choices": [{"text": "", "finish_reason": "length"}],
                "usage": {"completion_tokens": 1},
            },
            "[DONE]",
        ]
        with _FakeOpenAIServer(events) as (_server, url):
            result = stream_request_openai_completions(
                url=url,
                prompt=[1, 2, 3],
                max_tokens=2,
                req_id="openai-mismatch",
                timeout_s=5.0,
                model="gemma-test",
            )

        self.assertFalse(result["success"])
        self.assertEqual(result["output_tokens"], 1)
        self.assertIn("disagrees with usage", result["error"])

    def test_openai_truncated_stream_is_not_success(self) -> None:
        events = [
            {"choices": [{"token_ids": [101], "finish_reason": None}]},
            {"choices": [{"token_ids": [102], "finish_reason": None}]},
        ]
        with _FakeOpenAIServer(events) as (_server, url):
            result = stream_request_openai_completions(
                url=url,
                prompt=[1, 2, 3],
                max_tokens=2,
                req_id="openai-truncated",
                timeout_s=5.0,
                model="gemma-test",
            )

        self.assertFalse(result["success"])
        self.assertIn("without a finish event", result["error"])

    def test_extra_body_cannot_override_benchmark_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "benchmark-controlled fields"):
            validate_extra_body({"prompt": [9], "temperature": 1.0})
        validate_extra_body({"priority": 0})

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
        self.assertEqual(row["request_count"], 2)
        self.assertEqual(row["itl_valid_request_count"], 2)
        self.assertEqual(len(row["request_metrics"]), 2)

    def test_summarize_row_excludes_non_token_exact_itl(self) -> None:
        row = summarize_row(
            2,
            [
                {
                    "req_id": "exact",
                    "success": True,
                    "finish_reason": "length",
                    "error": None,
                    "output_tokens": 2,
                    "ttft_s": 0.10,
                    "e2e_latency_s": 0.30,
                    "decode_s": 0.20,
                    "itl_s": [0.20],
                    "itl_valid": True,
                    "output_token_count_exact": True,
                    "output_token_count_source": "token_ids",
                },
                {
                    "req_id": "chunked",
                    "success": True,
                    "finish_reason": "length",
                    "error": None,
                    "output_tokens": 2,
                    "ttft_s": 0.10,
                    "e2e_latency_s": 0.50,
                    "decode_s": 0.40,
                    "itl_s": [0.40],
                    "itl_valid": False,
                    "output_token_count_exact": True,
                    "output_token_count_source": "usage",
                },
            ],
            elapsed_s=0.5,
        )

        self.assertEqual(row["itl_valid_request_count"], 1)
        self.assertEqual(row["itl_sample_count"], 1)
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
            requests_per_row=3,
            synthetic_prompts=True,
            synthetic_vocab_size=128,
        )
        measured = build_prompts(args, row_offset=0)
        warmup = build_prompts(args, row_offset=64)

        for B in args.concurrency:
            self.assertEqual(len(measured[B]), 3)
            self.assertEqual([len(p) for p in measured[B]], [len(p) for p in warmup[B]])
            self.assertNotEqual(measured[B][0], warmup[B][0])
        self.assertTrue(
            {tuple(prompt) for prompt in measured[1]}.isdisjoint(
                {tuple(prompt) for prompt in measured[2]}
            )
        )

    def test_run_row_records_sustained_request_count_and_prompt_fingerprint(self) -> None:
        def fake_stream_request(**kwargs):
            return {
                "req_id": kwargs["req_id"],
                "success": True,
                "finish_reason": "length",
                "error": None,
                "output_tokens": kwargs["max_tokens"],
                "ttft_s": 0.01,
                "e2e_latency_s": 0.02,
                "decode_s": 0.01,
                "itl_s": [],
                "itl_valid": True,
                "output_token_count_exact": True,
                "output_token_count_source": "token_ids",
            }

        args = types.SimpleNamespace(
            backend="openai-completions",
            engine="test-engine",
            ctx=8,
            out=1,
            request_timeout_s=5.0,
            served_model="gemma-test",
            synthetic_prompts=True,
        )
        old_stream_request = bench.stream_request
        try:
            bench.stream_request = fake_stream_request
            row = run_row(
                "http://127.0.0.1:1",
                2,
                [[1, 2], [3, 4], [5, 6]],
                args,
                extra_body=None,
            )
        finally:
            bench.stream_request = old_stream_request

        self.assertEqual(row["B"], 2)
        self.assertEqual(row["request_count"], 3)
        self.assertEqual(row["success_count"], 3)
        self.assertEqual(row["prompt_count"], 3)
        self.assertEqual(row["prompt_total_tokens"], 6)
        self.assertEqual(row["prompt_token_source"], "synthetic")
        self.assertEqual(len(row["prompt_token_ids_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
