import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import threading
import time
import tempfile
import unittest
from unittest.mock import patch
import uuid

from experiments import gemma_multiturn_http_bench as bench
from experiments.gemma_multiturn_bench import (
    atomic_write_json,
    build_shared_history_trace,
    build_workload,
    load_shared_history_trace,
    shared_history_trace_payload,
)


def _trace_fixture(
    root: Path,
    *,
    sessions: int = 2,
    turns: int = 3,
    context_tokens: int = 4,
    delta_tokens: int = 2,
    output_tokens: int = 2,
    vocab_size: int = 512,
):
    workload = build_workload(
        sessions=sessions,
        turns=turns,
        initial_context_tokens=context_tokens,
        turn_input_tokens=delta_tokens,
        vocab_size=vocab_size,
    )
    turn_outputs = [
        [
            [
                100 + turn_index * 40 + session_index * 4 + token_index
                for token_index in range(output_tokens)
            ]
            for session_index in range(sessions)
        ]
        for turn_index in range(turns)
    ]
    path = root / "trace.json"
    trace = build_shared_history_trace(
        workload,
        turn_outputs,
        sessions=sessions,
        turns=turns,
        output_tokens_per_turn=output_tokens,
        vocab_size=vocab_size,
        source_path=str(path),
        source={"run_id": str(uuid.uuid4())},
    )
    atomic_write_json(path, shared_history_trace_payload(trace))
    return workload, trace, path


def _make_handler(
    *,
    style: str,
    trace_outputs=None,
    fail_request: tuple[int, int] | None = None,
):
    class MockHandler(BaseHTTPRequestHandler):
        records = []
        active = 0
        max_active = 0
        lock = threading.Lock()

        def log_message(self, *args) -> None:
            pass

        def _write_json(self, status: int, payload) -> None:
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _write_sse(self, events) -> None:
            payload = b"".join(
                (
                    f"data: {event}\n\n".encode()
                    if isinstance(event, str)
                    else f"data: {json.dumps(event)}\n\n".encode()
                )
                for event in events
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            request_id = str(
                body.get("req_id")
                or body.get("request_id")
                or body.get("rid")
                or self.headers.get("x-request-id")
            )
            match = re.search(r"-turn-(\d+)-session-(\d+)$", request_id)
            if match is None:
                self._write_json(400, {"error": "bad mock request id"})
                return
            turn_index = int(match.group(1))
            session_index = int(match.group(2))
            with type(self).lock:
                type(self).active += 1
                type(self).max_active = max(
                    type(self).max_active,
                    type(self).active,
                )
                type(self).records.append(
                    {
                        "path": self.path,
                        "body": body,
                        "turn": turn_index,
                        "session": session_index,
                    }
                )
            try:
                time.sleep(0.01)
                if fail_request == (turn_index, session_index):
                    self._write_json(500, {"error": "forced failure"})
                    return
                if style == "wkvm":
                    tokens = list(body["forced_output_ids"])
                elif style == "vllm-hook":
                    encoded = body["vllm_xargs"][
                        bench.TEACHER_FORCED_TOKEN_IDS_ARG
                    ]
                    tokens = json.loads(encoded)
                elif style == "sglang-hook":
                    tokens = list(
                        body["custom_params"][
                            bench.TEACHER_FORCED_TOKEN_IDS_ARG
                        ]
                    )
                elif style in {
                    "sglang-native-cumulative",
                    "sglang-native-incremental",
                }:
                    count = int(body["sampling_params"]["max_new_tokens"])
                    tokens = [
                        300 + turn_index * 40 + session_index * 4 + index
                        for index in range(count)
                    ]
                elif style == "autonomous":
                    count = int(body["max_tokens"])
                    tokens = [
                        300 + turn_index * 40 + session_index * 4 + index
                        for index in range(count)
                    ]
                else:
                    raise AssertionError(style)

                if style == "wkvm":
                    events = [
                        {"type": "queued", "req_id": request_id},
                        *[
                            {
                                "type": "token",
                                "req_id": request_id,
                                "index": index,
                                "token": token,
                            }
                            for index, token in enumerate(tokens)
                        ],
                        {
                            "type": "finish",
                            "req_id": request_id,
                            "finish_reason": "length",
                            "error": None,
                            "metrics": {
                                "reused_prefix_tokens": (
                                    0 if turn_index == 0 else 10
                                )
                            },
                        },
                    ]
                elif style in {
                    "sglang-native-cumulative",
                    "sglang-native-incremental",
                }:
                    events = []
                    for index, token in enumerate(tokens):
                        output_ids = (
                            tokens[: index + 1]
                            if style == "sglang-native-cumulative"
                            else [token]
                        )
                        events.append(
                            {
                                "output_ids": output_ids,
                                "meta_info": {
                                    "id": request_id,
                                    "completion_tokens": index + 1,
                                    "cached_tokens": (
                                        0 if turn_index == 0 else 7
                                    ),
                                    "finish_reason": (
                                        {"type": "length"}
                                        if index + 1 == len(tokens)
                                        else None
                                    ),
                                },
                            }
                        )
                    events.append("[DONE]")
                else:
                    events = []
                    for token in tokens:
                        choice = {
                            "index": 0,
                            "text": f"t{token}",
                            "finish_reason": None,
                        }
                        if style != "sglang-hook":
                            choice["token_ids"] = [token]
                        events.append(
                            {
                                "id": request_id,
                                "object": "text_completion",
                                "choices": [choice],
                            }
                        )
                    events.extend(
                        [
                            {
                                "id": request_id,
                                "object": "text_completion",
                                "choices": [
                                    {
                                        "index": 0,
                                        "text": "",
                                        "finish_reason": "length",
                                    }
                                ],
                            },
                            {
                                "id": request_id,
                                "object": "text_completion",
                                "choices": [],
                                "usage": {
                                    "prompt_tokens": len(body["prompt"]),
                                    "completion_tokens": len(tokens),
                                    "total_tokens": len(body["prompt"]) + len(tokens),
                                    "prompt_tokens_details": {
                                        "cached_tokens": (
                                            0 if turn_index == 0 else 7
                                        )
                                    },
                                },
                            },
                            "[DONE]",
                        ]
                    )
                self._write_sse(events)
            finally:
                with type(self).lock:
                    type(self).active -= 1

        def do_GET(self) -> None:
            if self.path != "/metrics":
                self._write_json(404, {"error": "not found"})
                return
            self._write_json(
                200,
                {
                    "server": {"token_sessions": 2},
                    "engine": {
                        "max_decode_batch_rows": 2,
                        "fallback_decode_model_calls": 0,
                    },
                },
            )

    return MockHandler


class _MockServer:
    def __init__(self, handler) -> None:
        self.handler = handler
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.thread = threading.Thread(
            target=self.server.serve_forever,
            daemon=True,
        )

    def __enter__(self):
        self.thread.start()
        host, port = self.server.server_address
        self.url = f"http://{host}:{port}"
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)


def _args(
    *,
    engine: str,
    url: str,
    output_json: Path,
    turns: int,
    trace_path: Path | None = None,
    write_trace_path: Path | None = None,
    extra=(),
):
    argv = [
        "--engine",
        engine,
        "--base-url",
        url,
        "--sessions",
        "2",
        "--turns",
        str(turns),
        "--initial-context-tokens",
        "4",
        "--turn-input-tokens",
        "2",
        "--output-tokens-per-turn",
        "2",
        "--synthetic-vocab-size",
        "512",
        "--request-timeout-s",
        "5",
        "--gpu-memory-device",
        "none",
        "--run-id",
        str(uuid.uuid4()),
        "--json",
        str(output_json),
    ]
    if engine != "wkvm":
        argv.extend(["--model", "mock-gemma"])
    if trace_path is not None:
        argv.extend(["--shared-history-trace-json", str(trace_path)])
    if write_trace_path is not None:
        argv.extend(["--write-shared-history-trace-json", str(write_trace_path)])
    argv.extend(extra)
    return bench.build_parser().parse_args(argv)


class GemmaMultiTurnHttpBenchTests(unittest.TestCase):
    def test_sse_buffer_preserves_json_across_every_split_point(self) -> None:
        event = {
            "output_ids": list(range(20)),
            "meta_info": {
                "text": "a deliberately fragmented string payload",
                "completion_tokens": 20,
            },
        }
        raw = f"data: {json.dumps(event)}\n\n".encode()
        for split in range(1, len(raw)):
            with self.subTest(split=split):
                decoder = bench._SSEEventBuffer()
                events = decoder.feed(raw[:split])
                events.extend(decoder.feed(raw[split:]))
                decoder.finish()
                self.assertEqual(events, [event])

    def test_sse_buffer_emits_multiple_combined_events_and_done(self) -> None:
        first = {"output_ids": [1], "meta_info": {"completion_tokens": 1}}
        second = {"output_ids": [2], "meta_info": {"completion_tokens": 2}}
        raw = (
            f"event: message\ndata: {json.dumps(first)}\n\n"
            f"data: {json.dumps(second)}\n\n"
            "data: [DONE]\n\n"
        ).encode()
        decoder = bench._SSEEventBuffer()
        self.assertEqual(decoder.feed(raw), [first, second, "[DONE]"])
        decoder.finish()

    def test_sse_buffer_rejects_non_whitespace_incomplete_eof(self) -> None:
        partial_line = bench._SSEEventBuffer()
        partial_line.feed(b'data: {"text":"unterminated')
        with self.assertRaisesRegex(ValueError, "incomplete SSE line at EOF"):
            partial_line.finish()

        partial_event = bench._SSEEventBuffer()
        partial_event.feed(b'data: {"complete":true}\n')
        with self.assertRaisesRegex(ValueError, "incomplete SSE event at EOF"):
            partial_event.finish()

    def test_default_workload_is_b16_eight_turn_ctx36k(self) -> None:
        args = bench.build_parser().parse_args(["--engine", "wkvm"])
        bench.validate_args(args)
        self.assertEqual(args.sessions, 16)
        self.assertEqual(args.turns, 8)
        self.assertEqual(args.initial_context_tokens, 36_864)
        self.assertEqual(args.turn_input_tokens, 32)
        self.assertEqual(args.output_tokens_per_turn, 64)
        self.assertEqual(args.endpoint, "/v1/stream")
        self.assertEqual(args.api_mode, "wkvm_token_native_session")

    def test_sglang_native_generate_defaults_and_validation(self) -> None:
        native_args = bench.build_parser().parse_args(
            [
                "--engine",
                "sglang",
                "--model",
                "mock-gemma",
                "--sglang-native-generate",
            ]
        )
        bench.validate_args(native_args)
        self.assertEqual(native_args.endpoint, "/generate")
        self.assertEqual(native_args.api_mode, "sglang_native_generate")

        openai_args = bench.build_parser().parse_args(
            ["--engine", "sglang", "--model", "mock-gemma"]
        )
        bench.validate_args(openai_args)
        self.assertEqual(openai_args.endpoint, "/v1/completions")
        self.assertEqual(openai_args.api_mode, "openai_completions")

        wrong_engine_args = bench.build_parser().parse_args(
            [
                "--engine",
                "vllm",
                "--model",
                "mock-gemma",
                "--sglang-native-generate",
            ]
        )
        with self.assertRaisesRegex(ValueError, "requires --engine sglang"):
            bench.validate_args(wrong_engine_args)

        replay_args = bench.build_parser().parse_args(
            [
                "--engine",
                "sglang",
                "--model",
                "mock-gemma",
                "--sglang-native-generate",
                "--shared-history-trace-json",
                "trace.json",
            ]
        )
        with self.assertRaisesRegex(ValueError, "does not support shared-history"):
            bench.validate_args(replay_args)

    def test_gpu_memory_options_require_valid_values_and_monitoring(self) -> None:
        invalid_cases = (
            (
                ("--gpu-memory-baseline-used-mib", "nan"),
                "baseline-used-mib must be finite",
            ),
            (("--memory-ceiling-mib", "0"), "memory-ceiling-mib must be finite"),
        )
        for extra, expected in invalid_cases:
            with self.subTest(extra=extra):
                args = bench.build_parser().parse_args(
                    ["--engine", "wkvm", *extra]
                )
                with self.assertRaisesRegex(ValueError, expected):
                    bench.validate_args(args)

        disabled_args = bench.build_parser().parse_args(
            [
                "--engine",
                "wkvm",
                "--gpu-memory-device",
                "none",
                "--gpu-memory-baseline-used-mib",
                "512",
            ]
        )
        with self.assertRaisesRegex(ValueError, "require --gpu-memory-device"):
            bench.validate_args(disabled_args)

    def test_prelaunch_memory_baseline_preserves_request_start_measurement(self) -> None:
        raw = {
            "schema": "wkvm.whole_gpu_memory.v1",
            "scope": "whole_device",
            "source": "nvidia-smi",
            "baseline_used_mib": 23_000,
            "peak_used_mib": 23_900,
            "peak_delta_mib": 900,
        }

        normalized = bench._apply_gpu_memory_contract(
            raw,
            prelaunch_baseline_used_mib=512.0,
            memory_ceiling_mib=24_200.0,
        )

        self.assertEqual(raw["baseline_used_mib"], 23_000)
        self.assertEqual(normalized["request_start_baseline_used_mib"], 23_000)
        self.assertEqual(normalized["request_start_baseline_source"], "nvidia-smi")
        self.assertEqual(
            normalized["request_start_baseline_scope"],
            "whole_device_request_start",
        )
        self.assertEqual(normalized["baseline_used_mib"], 512.0)
        self.assertEqual(
            normalized["baseline_source"],
            "operator_supplied_prelaunch_nvidia_smi",
        )
        self.assertEqual(
            normalized["baseline_scope"],
            "whole_device_pre_server_launch",
        )
        self.assertEqual(normalized["peak_used_mib"], 23_900)
        self.assertEqual(normalized["peak_delta_mib"], 23_388.0)
        self.assertEqual(normalized["memory_ceiling_mib"], 24_200.0)
        self.assertTrue(normalized["within_memory_ceiling"])

    def test_wkvm_replays_trace_with_stable_sessions_and_barriers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, trace, trace_path = _trace_fixture(root)
            handler = _make_handler(
                style="wkvm",
                trace_outputs=trace.turn_outputs,
            )
            with _MockServer(handler) as server:
                payload = bench.run(
                    _args(
                        engine="wkvm",
                        url=server.url,
                        output_json=root / "result.json",
                        turns=3,
                        trace_path=trace_path,
                        extra=("--server-metrics-url", f"{server.url}/metrics"),
                    )
                )

            self.assertEqual(payload["summary"]["success_count"], 6)
            self.assertEqual(payload["summary"]["error_count"], 0)
            self.assertEqual(
                payload["workload"]["history_policy"],
                "wkvm_token_session_initial_prompt_then_deltas",
            )
            self.assertEqual(
                payload["summary"]["continuation_turns"]["request_count"],
                4,
            )
            self.assertEqual(payload["turns"][1]["cached_tokens_total"], 20)
            self.assertEqual(payload["turns"][2]["p50_ttft_s"] is not None, True)
            self.assertEqual(payload["turns"][0]["generated_output_fingerprint"], trace.output_fingerprints[0])
            self.assertGreaterEqual(handler.max_active, 2)
            session_zero = sorted(
                (
                    record
                    for record in handler.records
                    if record["session"] == 0
                ),
                key=lambda record: record["turn"],
            )
            self.assertEqual(
                [
                    len(
                        record["body"].get(
                            "prompt_ids",
                            record["body"].get("delta_ids", []),
                        )
                    )
                    for record in session_zero
                ],
                [4, 2, 2],
            )
            self.assertIn("prompt_ids", session_zero[0]["body"])
            self.assertNotIn("delta_ids", session_zero[0]["body"])
            for record in session_zero[1:]:
                self.assertIn("delta_ids", record["body"])
                self.assertNotIn("prompt_ids", record["body"])
            for record in handler.records:
                self.assertEqual(
                    record["body"]["forced_output_ids"],
                    trace.turn_outputs[record["turn"]][record["session"]],
                )
            self.assertEqual(
                {record["body"]["session_id"] for record in session_zero},
                {"session-0000"},
            )
            self.assertEqual(
                payload["server_metrics_after_run"]["engine"][
                    "max_decode_batch_rows"
                ],
                2,
            )
            self.assertIsNone(payload["server_metrics_error"])
            self.assertTrue(payload["teacher_forcing_hook"]["enabled"])
            self.assertEqual(
                payload["teacher_forcing_hook"]["field_path"],
                "forced_output_ids",
            )

    def test_vllm_hook_uses_json_array_string_and_exact_response_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, trace, trace_path = _trace_fixture(root, turns=2)
            handler = _make_handler(style="vllm-hook")
            with _MockServer(handler) as server:
                payload = bench.run(
                    _args(
                        engine="vllm",
                        url=server.url,
                        output_json=root / "result.json",
                        turns=2,
                        trace_path=trace_path,
                    )
                )

            self.assertEqual(payload["summary"]["error_count"], 0)
            self.assertEqual(
                payload["teacher_forcing_hook"]["field_path"],
                "vllm_xargs.wkvm_teacher_forced_token_ids",
            )
            self.assertEqual(payload["teacher_forcing_hook"]["encoding"], "json-string")
            for record in handler.records:
                encoded = record["body"]["vllm_xargs"][
                    bench.TEACHER_FORCED_TOKEN_IDS_ARG
                ]
                self.assertIsInstance(encoded, str)
                self.assertEqual(
                    json.loads(encoded),
                    trace.turn_outputs[record["turn"]][record["session"]],
                )
            self.assertEqual(
                payload["turns"][1]["response_token_ids_observed_count"],
                2,
            )
            self.assertEqual(payload["turns"][1]["cached_tokens_total"], 14)

    def test_sglang_hook_uses_custom_params_and_count_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, trace, trace_path = _trace_fixture(root, turns=2)
            handler = _make_handler(style="sglang-hook")
            with _MockServer(handler) as server:
                payload = bench.run(
                    _args(
                        engine="sglang",
                        url=server.url,
                        output_json=root / "result.json",
                        turns=2,
                        trace_path=trace_path,
                        extra=(
                            "--teacher-forcing-processor",
                            "mock-serialized-processor",
                        ),
                    )
                )

            self.assertEqual(payload["summary"]["error_count"], 0)
            self.assertEqual(
                payload["teacher_forcing_hook"]["field_path"],
                "custom_params.wkvm_teacher_forced_token_ids",
            )
            self.assertEqual(payload["teacher_forcing_hook"]["encoding"], "array")
            for record in handler.records:
                self.assertEqual(
                    record["body"]["custom_params"][
                        bench.TEACHER_FORCED_TOKEN_IDS_ARG
                    ],
                    trace.turn_outputs[record["turn"]][record["session"]],
                )
                self.assertEqual(
                    record["body"]["custom_logit_processor"],
                    "mock-serialized-processor",
                )
            self.assertEqual(
                payload["turns"][1]["response_token_ids_observed_count"],
                0,
            )
            self.assertEqual(
                payload["turns"][1]["teacher_forcing"][
                    "hook_contract_verification_count"
                ],
                2,
            )
            self.assertEqual(
                payload["turns"][1]["generated_output_fingerprint"],
                trace.output_fingerprints[1],
            )

    def test_autonomous_openai_run_writes_reusable_trace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_trace_path = root / "source-trace.json"
            handler = _make_handler(style="autonomous")
            with _MockServer(handler) as server:
                payload = bench.run(
                    _args(
                        engine="vllm",
                        url=server.url,
                        output_json=root / "result.json",
                        turns=2,
                        write_trace_path=output_trace_path,
                    )
                )

            workload = build_workload(
                sessions=2,
                turns=2,
                initial_context_tokens=4,
                turn_input_tokens=2,
                vocab_size=512,
            )
            loaded = load_shared_history_trace(
                output_trace_path,
                workload,
                sessions=2,
                turns=2,
                output_tokens_per_turn=2,
                vocab_size=512,
            )
            self.assertEqual(payload["summary"]["error_count"], 0)
            self.assertEqual(
                payload["benchmark_identity"]["artifact_role"],
                "http_trace_source",
            )
            self.assertEqual(len(loaded.turn_outputs), 2)
            session_zero = sorted(
                (
                    record
                    for record in handler.records
                    if record["session"] == 0
                ),
                key=lambda record: record["turn"],
            )
            self.assertEqual(
                [len(record["body"]["prompt"]) for record in session_zero],
                [4, 8],
            )

    def test_sglang_native_generate_writes_exact_trace_for_both_stream_modes(
        self,
    ) -> None:
        for style in (
            "sglang-native-cumulative",
            "sglang-native-incremental",
        ):
            with self.subTest(style=style), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output_trace_path = root / "source-trace.json"
                handler = _make_handler(style=style)
                with _MockServer(handler) as server:
                    payload = bench.run(
                        _args(
                            engine="sglang",
                            url=server.url,
                            output_json=root / "result.json",
                            turns=2,
                            write_trace_path=output_trace_path,
                            extra=("--sglang-native-generate",),
                        )
                    )

                workload = build_workload(
                    sessions=2,
                    turns=2,
                    initial_context_tokens=4,
                    turn_input_tokens=2,
                    vocab_size=512,
                )
                loaded = load_shared_history_trace(
                    output_trace_path,
                    workload,
                    sessions=2,
                    turns=2,
                    output_tokens_per_turn=2,
                    vocab_size=512,
                )
                self.assertEqual(payload["summary"]["error_count"], 0)
                self.assertEqual(payload["api"]["mode"], "sglang_native_generate")
                self.assertEqual(payload["api"]["endpoint"], "/generate")
                self.assertEqual(
                    payload["benchmark_identity"]["artifact_role"],
                    "http_trace_source",
                )
                self.assertEqual(payload["turns"][1]["cached_tokens_total"], 14)
                self.assertEqual(
                    [
                        turn["response_token_ids_observed_count"]
                        for turn in payload["turns"]
                    ],
                    [2, 2],
                )
                self.assertTrue(
                    all(
                        request["stream_exact_token_signals"] == 2
                        for turn in payload["turns"]
                        for request in turn["requests"]
                    )
                )
                self.assertEqual(
                    loaded.turn_outputs,
                    [
                        [[300, 301], [304, 305]],
                        [[340, 341], [344, 345]],
                    ],
                )
                session_zero = sorted(
                    (
                        record
                        for record in handler.records
                        if record["session"] == 0
                    ),
                    key=lambda record: record["turn"],
                )
                self.assertEqual(
                    [len(record["body"]["input_ids"]) for record in session_zero],
                    [4, 8],
                )
                for record in handler.records:
                    body = record["body"]
                    self.assertEqual(record["path"], "/generate")
                    self.assertTrue(body["stream"])
                    self.assertIn("-turn-", body["rid"])
                    self.assertNotIn("prompt", body)
                    self.assertNotIn("model", body)
                    self.assertEqual(
                        body["sampling_params"],
                        {
                            "temperature": 0.0,
                            "top_p": 1.0,
                            "max_new_tokens": 2,
                            "ignore_eos": True,
                        },
                    )

    def test_http_error_is_recorded_without_losing_later_trace_turns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _, trace, trace_path = _trace_fixture(root)
            handler = _make_handler(
                style="wkvm",
                trace_outputs=trace.turn_outputs,
                fail_request=(1, 1),
            )
            with _MockServer(handler) as server:
                payload = bench.run(
                    _args(
                        engine="wkvm",
                        url=server.url,
                        output_json=root / "result.json",
                        turns=3,
                        trace_path=trace_path,
                    )
                )

            self.assertEqual(payload["summary"]["completed_turn_rows"], 3)
            self.assertEqual(payload["summary"]["error_count"], 1)
            self.assertEqual(payload["turns"][1]["error_count"], 1)
            self.assertIn("forced failure", payload["turns"][1]["errors"][0]["error"])
            self.assertEqual(payload["turns"][2]["success_count"], 2)

    def test_memory_ceiling_failure_writes_artifact_before_raising(self) -> None:
        class FakeMemoryMonitor:
            def __init__(self, device, interval_s) -> None:
                self.device = device
                self.interval_s = interval_s

            def __enter__(self):
                return self

            def __exit__(self, *exc) -> None:
                pass

            def result(self):
                return {
                    "schema": "wkvm.whole_gpu_memory.v1",
                    "scope": "whole_device",
                    "source": "nvidia-smi",
                    "device_selector": self.device,
                    "device_index": 0,
                    "device_uuid": "GPU-test",
                    "gpu_name": "NVIDIA GeForce RTX 4090",
                    "driver_version": "test",
                    "memory_total_mib": 24_564,
                    "sample_interval_s": self.interval_s,
                    "sample_count": 2,
                    "baseline_used_mib": 23_000,
                    "peak_used_mib": 24_350,
                    "peak_delta_mib": 1_350,
                    "query_error_count": 0,
                    "error": None,
                }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_path = root / "over-ceiling.json"
            handler = _make_handler(style="autonomous")
            with _MockServer(handler) as server:
                args = _args(
                    engine="vllm",
                    url=server.url,
                    output_json=output_path,
                    turns=1,
                    extra=(
                        "--gpu-memory-device",
                        "0",
                        "--gpu-memory-baseline-used-mib",
                        "512",
                        "--memory-ceiling-mib",
                        "24200",
                    ),
                )
                with patch.object(
                    bench,
                    "WholeGpuMemoryMonitor",
                    FakeMemoryMonitor,
                ), patch.object(
                    bench,
                    "collect_gpu_provenance",
                    return_value=(None, None),
                ), self.assertRaisesRegex(
                    RuntimeError,
                    "24350.0 MiB exceeds ceiling 24200.0 MiB",
                ):
                    bench.run(args)

            payload = json.loads(output_path.read_text())
            memory = payload["gpu_memory"]
            self.assertEqual(memory["request_start_baseline_used_mib"], 23_000)
            self.assertEqual(memory["baseline_used_mib"], 512.0)
            self.assertEqual(memory["peak_delta_mib"], 23_838.0)
            self.assertFalse(memory["within_memory_ceiling"])
            self.assertEqual(
                payload["benchmark_identity"]["memory_ceiling_mib"],
                24_200.0,
            )
            self.assertEqual(payload["fatal_error"]["phase"], "gpu_memory_ceiling")
            monitor = payload["provenance"]["gpu_memory_monitor"]
            self.assertEqual(monitor["baseline_scope"], "whole_device_pre_server_launch")
            self.assertFalse(monitor["within_memory_ceiling"])


if __name__ == "__main__":
    unittest.main()
