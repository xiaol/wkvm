import asyncio
import sys
import types
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from experiments.gemma_multiturn_bench import (
    _force_pending_wkvm_outputs,
    _teacher_forcing_errors,
    _turn_prompts_and_deltas,
    atomic_write_json,
    build_parser,
    build_payload,
    build_shared_history_trace,
    build_workload,
    extract_sglang_cached_tokens,
    extract_vllm_cached_tokens,
    load_shared_history_trace,
    percentile,
    request_order_indices,
    restore_logical_order,
    run_sglang,
    run_vllm,
    shared_history_trace_metadata,
    shared_history_trace_payload,
    summarize_run,
    summarize_turn,
    validate_args,
    workload_fingerprints,
)


def _fake_vllm_modules():
    vllm = types.ModuleType("vllm")

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeVLLM:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            type(self).instances.append(self)

        def generate(self, requests, sampling, use_tqdm=False):
            del sampling, use_tqdm
            return [
                types.SimpleNamespace(
                    outputs=[types.SimpleNamespace(token_ids=[41, 42])],
                    metrics=None,
                    num_cached_tokens=0,
                )
                for _ in requests
            ]

    vllm.LLM = FakeVLLM
    vllm.SamplingParams = FakeSamplingParams
    vllm.__version__ = "fake-vllm"

    incumbent = types.ModuleType("incumbent_gemma_bench")
    incumbent.cleanup_cuda = lambda: None
    incumbent.synchronize_cuda = lambda: None
    incumbent.vllm_capacity_telemetry = lambda llm, max_model_len: {
        "kv_token_capacity": 1_024,
        "kv_max_concurrency": 2.0,
        "capacity_source": "fake",
        "capacity_estimated": False,
    }
    return {
        "vllm": vllm,
        "incumbent_gemma_bench": incumbent,
    }, FakeVLLM


def _fake_sglang_modules(engine_type):
    sglang = types.ModuleType("sglang")
    sglang.Engine = engine_type
    sglang.__version__ = "fake-sglang"

    incumbent = types.ModuleType("incumbent_gemma_bench")
    incumbent.cleanup_cuda = lambda: None
    incumbent.sglang_capacity_telemetry = lambda engine: {
        "fake_capacity": True
    }
    incumbent.sglang_language_model_override = lambda model_path: {
        "model_path": model_path
    }
    incumbent.synchronize_cuda = lambda: None

    logits = types.ModuleType("experiments.sglang_shared_history_logits")

    class FakeSharedHistoryLogitsProcessor:
        @classmethod
        def to_str(cls):
            return "fake-teacher-forcing-processor"

    logits.SharedHistoryLogitsProcessor = FakeSharedHistoryLogitsProcessor
    return {
        "sglang": sglang,
        "incumbent_gemma_bench": incumbent,
        "experiments.sglang_shared_history_logits": logits,
    }


def _fake_sglang_engine_type(
    *,
    open_failure_session=None,
    generate_failure_session=None,
    close_failure_session=None,
):
    class FakeSGLangEngine:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.loop = asyncio.new_event_loop()
            self.open_calls = []
            self.generate_calls = []
            self.batch_generate_calls = []
            self.close_calls = []
            self.session_turns = {}
            self.active_requests = 0
            self.max_active_requests = 0
            self.shutdown_called = False
            type(self).instances.append(self)

        def open_session(
            self,
            capacity_of_str_len,
            session_id=None,
            streaming=False,
        ):
            self.open_calls.append(
                (capacity_of_str_len, session_id, streaming)
            )
            if session_id == open_failure_session:
                return None
            return session_id

        def generate(self, **kwargs):
            self.batch_generate_calls.append(kwargs)
            input_ids = kwargs["input_ids"]
            sampling_params = kwargs["sampling_params"]
            if not isinstance(sampling_params, list):
                sampling_params = [sampling_params] * len(input_ids)
            outputs = []
            for index, sampling in enumerate(sampling_params):
                output_ids = list(
                    sampling.get("custom_params", {}).get(
                        "wkvm_teacher_forced_token_ids",
                        [41, 42],
                    )
                )
                outputs.append(
                    {
                        "meta_info": {
                            "output_ids": output_ids,
                            "id": f"batch-rid-{index}",
                            "cached_tokens": len(input_ids[index]),
                        }
                    }
                )
            return outputs

        async def async_generate(self, **kwargs):
            session_params = kwargs["session_params"]
            native_session_id = session_params["id"]
            turn_index = self.session_turns.get(native_session_id, 0)
            self.session_turns[native_session_id] = turn_index + 1
            self.generate_calls.append(kwargs)
            self.active_requests += 1
            self.max_active_requests = max(
                self.max_active_requests,
                self.active_requests,
            )
            try:
                await asyncio.sleep(0)
                if native_session_id == generate_failure_session:
                    raise RuntimeError("fake generate failure")
                sampling = kwargs["sampling_params"]
                output_ids = list(
                    sampling.get("custom_params", {}).get(
                        "wkvm_teacher_forced_token_ids",
                        [41, 42],
                    )
                )
                return {
                    "meta_info": {
                        "output_ids": output_ids,
                        "id": f"{native_session_id}-rid-{turn_index}",
                        "cached_tokens": 10 + turn_index,
                        "first_token_latency": 0.01,
                        "e2e_latency": 0.02,
                    }
                }
            finally:
                self.active_requests -= 1

        def close_session(self, native_session_id):
            self.close_calls.append(native_session_id)
            if native_session_id == close_failure_session:
                raise RuntimeError("fake close failure")

        def shutdown(self):
            self.shutdown_called = True
            self.loop.close()

    return FakeSGLangEngine


class TestGemmaMultiturnBench(unittest.TestCase):
    def test_default_history_mode_remains_engine_generated(self) -> None:
        args = build_parser().parse_args(["--engine", "wkvm"])

        self.assertIsNone(args.shared_history_trace_json)
        self.assertIsNone(args.write_shared_history_trace_json)
        self.assertFalse(args.sglang_streaming_session)
        self.assertIsNone(args.sglang_streaming_session_capacity)
        self.assertIsNone(args.campaign_id)
        self.assertIsNone(args.repeat_id)
        self.assertIsNone(args.run_id)

    def test_campaign_identity_is_validated_and_run_id_is_generated(self) -> None:
        args = build_parser().parse_args(
            [
                "--engine",
                "wkvm",
                "--campaign-id",
                "campaign-1",
                "--repeat-id",
                "r1",
                "--memory-ceiling-mib",
                "23934",
            ]
        )

        validate_args(args)

        self.assertEqual(args.campaign_id, "campaign-1")
        self.assertEqual(args.repeat_id, "r1")
        self.assertEqual(len(args.run_id), 36)
        self.assertEqual(args.memory_ceiling_mib, 23_934)

    def test_campaign_identity_must_be_complete(self) -> None:
        args = build_parser().parse_args(
            ["--engine", "wkvm", "--campaign-id", "campaign-1"]
        )

        with self.assertRaisesRegex(ValueError, "supplied together"):
            validate_args(args)

    def test_sglang_streaming_sessions_submit_initial_then_deltas_concurrently(
        self,
    ) -> None:
        workload = build_workload(
            sessions=2,
            turns=2,
            initial_context_tokens=12,
            turn_input_tokens=3,
            vocab_size=64,
        )
        args = build_parser().parse_args(
            [
                "--engine",
                "sglang",
                "--sessions",
                "2",
                "--turns",
                "2",
                "--initial-context-tokens",
                "12",
                "--turn-input-tokens",
                "3",
                "--output-tokens-per-turn",
                "2",
                "--synthetic-vocab-size",
                "64",
                "--sglang-streaming-session",
                "--sglang-streaming-session-capacity",
                "32",
                "--sglang-chunked-prefill-size",
                "8192",
            ]
        )
        validate_args(args)
        args.model_path = "/models/gemma-test"
        engine_type = _fake_sglang_engine_type()

        with mock.patch.dict(
            sys.modules,
            _fake_sglang_modules(engine_type),
        ):
            result = run_sglang(args, workload)

        engine = engine_type.instances[-1]
        self.assertTrue(engine.kwargs["enable_streaming_session"])
        self.assertEqual(engine.kwargs["chunked_prefill_size"], 8_192)
        self.assertEqual(
            result["engine_config"]["chunked_prefill_size"],
            8_192,
        )
        self.assertEqual(
            engine.open_calls,
            [
                (32, "session-0000", True),
                (32, "session-0001", True),
            ],
        )
        self.assertEqual(engine.max_active_requests, 2)
        self.assertEqual(len(engine.generate_calls), 4)
        calls_by_session = {
            native_session_id: [
                call
                for call in engine.generate_calls
                if call["session_params"]["id"] == native_session_id
            ]
            for native_session_id in ("session-0000", "session-0001")
        }
        for index, native_session_id in enumerate(calls_by_session):
            first, second = calls_by_session[native_session_id]
            self.assertEqual(first["input_ids"], workload.initial_prompts[index])
            self.assertEqual(
                second["input_ids"],
                workload.turn_deltas[0][index],
            )
            self.assertIsNone(first["session_params"]["rid"])
            self.assertEqual(
                second["session_params"]["rid"],
                f"{native_session_id}-rid-0",
            )
            self.assertNotIn("custom_logit_processor", first)
        self.assertEqual(
            engine.close_calls,
            ["session-0000", "session-0001"],
        )
        self.assertTrue(engine.shutdown_called)
        telemetry = result["engine_config"]["session_telemetry"]
        self.assertTrue(telemetry["all_sessions_opened"])
        self.assertTrue(telemetry["all_opened_sessions_closed"])
        self.assertEqual(telemetry["generate_request_count"], 4)
        self.assertEqual(telemetry["generate_failure_count"], 0)
        self.assertEqual(
            result["turns"][1]["reuse_kind"],
            "sglang_streaming_session_append",
        )
        self.assertEqual(result["turns"][1]["history_submission"], "token_delta")
        self.assertEqual(result["turns"][1]["submitted_input_tokens"], 6)
        self.assertEqual(result["turns"][1]["error_count"], 0)

    def test_sglang_shared_trace_replay_requires_native_source(self) -> None:
        workload = build_workload(
            sessions=1,
            turns=1,
            initial_context_tokens=12,
            turn_input_tokens=3,
            vocab_size=64,
        )
        trace = build_shared_history_trace(
            workload,
            [[[10, 11]]],
            sessions=1,
            turns=1,
            output_tokens_per_turn=2,
            vocab_size=64,
        )
        args = build_parser().parse_args(
            [
                "--engine",
                "sglang",
                "--sessions",
                "1",
                "--turns",
                "1",
                "--initial-context-tokens",
                "12",
                "--turn-input-tokens",
                "3",
                "--output-tokens-per-turn",
                "2",
                "--synthetic-vocab-size",
                "64",
            ]
        )

        with self.assertRaisesRegex(RuntimeError, "native trace source"):
            run_sglang(args, workload, trace)

    def test_vllm_prefill_budget_is_passed_and_recorded(self) -> None:
        workload = build_workload(
            sessions=1,
            turns=1,
            initial_context_tokens=12,
            turn_input_tokens=3,
            vocab_size=64,
        )
        args = build_parser().parse_args(
            [
                "--engine",
                "vllm",
                "--sessions",
                "1",
                "--turns",
                "1",
                "--initial-context-tokens",
                "12",
                "--output-tokens-per-turn",
                "2",
                "--synthetic-vocab-size",
                "64",
                "--vllm-max-num-batched-tokens",
                "16384",
            ]
        )
        validate_args(args)
        args.model_path = "/models/gemma-test"
        modules, engine_type = _fake_vllm_modules()

        with mock.patch.dict(sys.modules, modules):
            result = run_vllm(args, workload)

        self.assertEqual(
            engine_type.instances[-1].kwargs["max_num_batched_tokens"],
            16_384,
        )
        self.assertEqual(
            result["engine_config"]["max_num_batched_tokens"],
            16_384,
        )

    def test_sglang_default_remains_native_radix_batching(self) -> None:
        workload = build_workload(
            sessions=2,
            turns=2,
            initial_context_tokens=12,
            turn_input_tokens=3,
            vocab_size=64,
        )
        args = build_parser().parse_args(
            [
                "--engine",
                "sglang",
                "--sessions",
                "2",
                "--turns",
                "2",
                "--initial-context-tokens",
                "12",
                "--turn-input-tokens",
                "3",
                "--output-tokens-per-turn",
                "2",
                "--synthetic-vocab-size",
                "64",
            ]
        )
        validate_args(args)
        args.model_path = "/models/gemma-test"
        engine_type = _fake_sglang_engine_type()

        with mock.patch.dict(
            sys.modules,
            _fake_sglang_modules(engine_type),
        ):
            result = run_sglang(args, workload)

        engine = engine_type.instances[-1]
        self.assertNotIn("enable_streaming_session", engine.kwargs)
        self.assertEqual(engine.open_calls, [])
        self.assertEqual(engine.generate_calls, [])
        self.assertEqual(len(engine.batch_generate_calls), 2)
        self.assertEqual(
            [len(prompt) for prompt in engine.batch_generate_calls[1]["input_ids"]],
            [17, 17],
        )
        self.assertEqual(result["turns"][1]["reuse_kind"], "sglang_radix_cache")
        self.assertFalse(result["engine_config"]["session_telemetry"]["enabled"])

    def test_sglang_streaming_session_records_generate_and_close_failures(
        self,
    ) -> None:
        workload = build_workload(
            sessions=2,
            turns=1,
            initial_context_tokens=12,
            turn_input_tokens=3,
            vocab_size=64,
        )
        args = build_parser().parse_args(
            [
                "--engine",
                "sglang",
                "--sessions",
                "2",
                "--turns",
                "1",
                "--initial-context-tokens",
                "12",
                "--output-tokens-per-turn",
                "2",
                "--synthetic-vocab-size",
                "64",
                "--sglang-streaming-session",
            ]
        )
        validate_args(args)
        args.model_path = "/models/gemma-test"
        engine_type = _fake_sglang_engine_type(
            generate_failure_session="session-0001",
            close_failure_session="session-0000",
        )

        with mock.patch.dict(
            sys.modules,
            _fake_sglang_modules(engine_type),
        ):
            result = run_sglang(args, workload)

        telemetry = result["engine_config"]["session_telemetry"]
        self.assertEqual(telemetry["generate_failure_count"], 1)
        self.assertEqual(telemetry["close_attempt_count"], 2)
        self.assertEqual(telemetry["close_failure_count"], 1)
        self.assertFalse(telemetry["all_opened_sessions_closed"])
        self.assertEqual(result["turns"][0]["error_count"], 1)
        self.assertIn(
            "fake generate failure",
            result["turns"][0]["errors"][0]["error"],
        )
        self.assertTrue(engine_type.instances[-1].shutdown_called)

    def test_sglang_streaming_session_open_failure_prevents_partial_turn(
        self,
    ) -> None:
        workload = build_workload(
            sessions=2,
            turns=1,
            initial_context_tokens=12,
            turn_input_tokens=3,
            vocab_size=64,
        )
        args = build_parser().parse_args(
            [
                "--engine",
                "sglang",
                "--sessions",
                "2",
                "--turns",
                "1",
                "--initial-context-tokens",
                "12",
                "--output-tokens-per-turn",
                "2",
                "--synthetic-vocab-size",
                "64",
                "--sglang-streaming-session",
            ]
        )
        validate_args(args)
        args.model_path = "/models/gemma-test"
        engine_type = _fake_sglang_engine_type(
            open_failure_session="session-0001",
        )

        with mock.patch.dict(
            sys.modules,
            _fake_sglang_modules(engine_type),
        ):
            result = run_sglang(args, workload)

        engine = engine_type.instances[-1]
        telemetry = result["engine_config"]["session_telemetry"]
        self.assertEqual(result["turns"], [])
        self.assertEqual(engine.generate_calls, [])
        self.assertEqual(telemetry["open_failure_count"], 1)
        self.assertFalse(telemetry["all_sessions_opened"])
        self.assertEqual(engine.close_calls, ["session-0000"])

    def test_incumbent_prefill_tuning_args_are_explicit(self) -> None:
        args = build_parser().parse_args(
            [
                "--engine",
                "vllm",
                "--vllm-max-num-batched-tokens",
                "16384",
                "--sglang-chunked-prefill-size",
                "8192",
            ]
        )

        self.assertEqual(args.vllm_max_num_batched_tokens, 16_384)
        self.assertEqual(args.sglang_chunked_prefill_size, 8_192)

    def test_incumbent_prefill_tuning_args_must_be_positive(self) -> None:
        args = build_parser().parse_args(
            [
                "--engine",
                "vllm",
                "--vllm-max-num-batched-tokens",
                "0",
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            "--vllm-max-num-batched-tokens must be >= 1",
        ):
            validate_args(args)

    def test_sglang_chunked_prefill_can_be_disabled_explicitly(self) -> None:
        args = build_parser().parse_args(
            [
                "--engine",
                "sglang",
                "--sglang-chunked-prefill-size",
                "-1",
            ]
        )

        validate_args(args)
        self.assertEqual(args.sglang_chunked_prefill_size, -1)

    def test_request_order_policies_are_deterministic_permutations(self) -> None:
        self.assertEqual(request_order_indices(4, 0, "forward"), [0, 1, 2, 3])
        self.assertEqual(request_order_indices(4, 0, "alternating"), [0, 1, 2, 3])
        self.assertEqual(request_order_indices(4, 1, "alternating"), [3, 2, 1, 0])
        shuffled = request_order_indices(8, 2, "seeded-shuffle", seed=17)
        self.assertEqual(sorted(shuffled), list(range(8)))
        self.assertEqual(
            shuffled,
            request_order_indices(8, 2, "seeded-shuffle", seed=17),
        )
        self.assertEqual(
            restore_logical_order(["d", "c", "b", "a"], [3, 2, 1, 0]),
            ["a", "b", "c", "d"],
        )

    def test_workload_is_deterministic_distinct_and_exact_length(self) -> None:
        kwargs = {
            "sessions": 3,
            "turns": 2,
            "initial_context_tokens": 96,
            "turn_input_tokens": 5,
            "vocab_size": 128,
        }
        first = build_workload(**kwargs)
        second = build_workload(**kwargs)

        self.assertEqual(first, second)
        self.assertEqual([len(prompt) for prompt in first.initial_prompts], [96] * 3)
        self.assertTrue(all(prompt[0] == 2 for prompt in first.initial_prompts))
        self.assertEqual(len({tuple(prompt) for prompt in first.initial_prompts}), 3)
        self.assertEqual(len(first.turn_deltas), 1)
        for deltas in first.turn_deltas:
            self.assertEqual([len(delta) for delta in deltas], [5] * 3)
            self.assertEqual(len({tuple(delta) for delta in deltas}), 3)

        fingerprints = workload_fingerprints(first)
        self.assertEqual(
            fingerprints["initial_prompts"]["prompt_total_tokens"],
            288,
        )
        self.assertEqual(
            [row["prompt_total_tokens"] for row in fingerprints["turn_deltas"]],
            [15],
        )
        self.assertEqual(
            len(fingerprints["initial_prompts"]["prompt_token_ids_sha256"]),
            64,
        )

    def test_shared_history_trace_round_trip_is_content_addressed(self) -> None:
        workload = build_workload(
            sessions=2,
            turns=3,
            initial_context_tokens=12,
            turn_input_tokens=3,
            vocab_size=64,
        )
        turn_outputs = [
            [[10, 11], [12, 13]],
            [[14, 15], [16, 17]],
            [[18, 19], [20, 21]],
        ]
        trace = build_shared_history_trace(
            workload,
            turn_outputs,
            sessions=2,
            turns=3,
            output_tokens_per_turn=2,
            vocab_size=64,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            trace_path = Path(temporary_directory) / "trace.json"
            source = {
                "campaign_id": "campaign-1",
                "repeat_id": "r1",
                "run_id": "11111111-1111-4111-8111-111111111111",
                "memory_ceiling_mib": 23_934.0,
            }
            atomic_write_json(
                trace_path,
                shared_history_trace_payload(trace, source=source),
            )
            loaded = load_shared_history_trace(
                trace_path,
                workload,
                sessions=2,
                turns=3,
                output_tokens_per_turn=2,
                vocab_size=64,
            )

            self.assertEqual(loaded.trace_sha256, trace.trace_sha256)
            self.assertEqual(loaded.turn_outputs, turn_outputs)
            self.assertEqual(loaded.output_fingerprints, trace.output_fingerprints)
            self.assertEqual(loaded.source, source)

            payload = shared_history_trace_payload(trace)
            payload["turn_outputs"][1][0][0] = 22
            atomic_write_json(trace_path, payload)
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_shared_history_trace(
                    trace_path,
                    workload,
                    sessions=2,
                    turns=3,
                    output_tokens_per_turn=2,
                    vocab_size=64,
                )

    def test_shared_trace_metadata_is_embedded_in_benchmark_artifact(self) -> None:
        workload = build_workload(
            sessions=2,
            turns=2,
            initial_context_tokens=12,
            turn_input_tokens=3,
            vocab_size=64,
        )
        trace = build_shared_history_trace(
            workload,
            [[[10, 11], [12, 13]], [[14, 15], [16, 17]]],
            sessions=2,
            turns=2,
            output_tokens_per_turn=2,
            vocab_size=64,
            source={
                "run_id": "11111111-1111-4111-8111-111111111111",
            },
        )
        args = build_parser().parse_args(
            [
                "--engine",
                "wkvm",
                "--sessions",
                "2",
                "--turns",
                "2",
                "--initial-context-tokens",
                "12",
                "--turn-input-tokens",
                "3",
                "--output-tokens-per-turn",
                "2",
                "--synthetic-vocab-size",
                "64",
            ]
        )
        args.required_model_len = 19
        args.model_path = "/models/gemma-test"
        args.campaign_id = "campaign-1"
        args.repeat_id = "r1"
        args.run_id = "22222222-2222-4222-8222-222222222222"
        args.memory_ceiling_mib = 23_934.0

        payload = build_payload(
            args,
            workload,
            {"turns": [], "engine_config": {}},
            {},
            shared_history_trace=trace,
        )

        self.assertEqual(
            payload["workload"]["history_policy"],
            "shared_teacher_forced_token_history",
        )
        self.assertEqual(
            payload["workload"]["history_trace_sha256"],
            trace.trace_sha256,
        )
        self.assertEqual(
            payload["workload"]["fingerprints"]["teacher_forced_turn_outputs"],
            trace.output_fingerprints,
        )
        self.assertTrue(payload["history_trace"]["teacher_forced"])
        self.assertEqual(
            payload["benchmark_identity"],
            {
                "campaign_id": "campaign-1",
                "repeat_id": "r1",
                "run_id": "22222222-2222-4222-8222-222222222222",
                "source_run_id": "11111111-1111-4111-8111-111111111111",
                "artifact_role": "teacher_forced_replay",
                "memory_ceiling_mib": 23_934.0,
            },
        )

    def test_emitted_trace_metadata_keeps_native_outputs_as_source_proof(self) -> None:
        workload = build_workload(
            sessions=2,
            turns=2,
            initial_context_tokens=12,
            turn_input_tokens=3,
            vocab_size=64,
        )
        trace = build_shared_history_trace(
            workload,
            [[[10, 11], [12, 13]], [[14, 15], [16, 17]]],
            sessions=2,
            turns=2,
            output_tokens_per_turn=2,
            vocab_size=64,
        )
        args = build_parser().parse_args(
            [
                "--engine",
                "wkvm",
                "--sessions",
                "2",
                "--turns",
                "2",
                "--initial-context-tokens",
                "12",
                "--turn-input-tokens",
                "3",
                "--output-tokens-per-turn",
                "2",
                "--synthetic-vocab-size",
                "64",
            ]
        )
        args.required_model_len = 19
        args.model_path = "/models/gemma-test"
        turns = [
            {
                "turn_index": turn_index,
                "generated_output_fingerprint": output_fingerprint,
                "request_output_token_ids_sha256": output_fingerprint[
                    "request_output_token_ids_sha256"
                ],
                "teacher_forcing": {
                    "enabled": False,
                    "mode": "engine_generated",
                },
            }
            for turn_index, output_fingerprint in enumerate(
                trace.output_fingerprints
            )
        ]

        payload = build_payload(
            args,
            workload,
            {
                "turns": turns,
                "engine_config": {
                    "history_mode": "engine_generated",
                    "teacher_forcing_backend": None,
                    "teacher_forcing_overhead_contract": None,
                },
            },
            {},
            emitted_history_trace=shared_history_trace_metadata(trace),
        )

        self.assertEqual(payload["history_trace"]["mode"], "engine_generated")
        self.assertFalse(payload["sampling"]["teacher_forced"])
        self.assertEqual(
            payload["emitted_history_trace"]["trace_sha256"],
            trace.trace_sha256,
        )
        self.assertEqual(
            [turn["generated_output_fingerprint"] for turn in payload["turns"]],
            payload["emitted_history_trace"]["output_fingerprints"],
        )

    def test_wkvm_pending_tokens_are_forced_before_next_cache_step(self) -> None:
        requests = {
            "session-0000": types.SimpleNamespace(output_token_ids=[91]),
            "session-0001": types.SimpleNamespace(output_token_ids=[92]),
        }
        candidates = [[], []]

        _force_pending_wkvm_outputs(
            requests,
            ["session-0000", "session-0001"],
            [[11, 12], [21, 22]],
            candidates,
        )
        requests["session-0000"].output_token_ids.append(93)
        requests["session-0001"].output_token_ids.append(94)
        _force_pending_wkvm_outputs(
            requests,
            ["session-0000", "session-0001"],
            [[11, 12], [21, 22]],
            candidates,
        )

        self.assertEqual(candidates, [[91, 93], [92, 94]])
        self.assertEqual(requests["session-0000"].output_token_ids, [11, 12])
        self.assertEqual(requests["session-0001"].output_token_ids, [21, 22])

    def test_teacher_forcing_rejects_a_backend_that_ignores_trace(self) -> None:
        errors = _teacher_forcing_errors(
            [[11, 12], [21, 99]],
            [[11, 12], [21, 22]],
            2,
        )

        self.assertEqual(errors[0], None)
        self.assertIn("diverged from shared teacher trace", errors[1])
        self.assertIn("index 1: expected 22, got 99", errors[1])

    def test_turn_zero_uses_exact_initial_prompt_before_continuation_deltas(self) -> None:
        workload = build_workload(
            sessions=2,
            turns=3,
            initial_context_tokens=12,
            turn_input_tokens=3,
            vocab_size=64,
        )
        histories = [list(prompt) for prompt in workload.initial_prompts]

        prompts, deltas = _turn_prompts_and_deltas(workload, histories, 0)
        self.assertEqual(prompts, workload.initial_prompts)
        self.assertEqual(deltas, [[], []])
        self.assertEqual(histories, workload.initial_prompts)

        prompts, deltas = _turn_prompts_and_deltas(workload, histories, 1)
        self.assertEqual(deltas, workload.turn_deltas[0])
        self.assertEqual([len(prompt) for prompt in prompts], [15, 15])

    def test_summarize_turn_reports_accounting_percentiles_and_fingerprints(self) -> None:
        row = summarize_turn(
            turn_index=0,
            session_ids=["session-0000", "session-0001"],
            prompts=[list(range(10)), list(range(10, 20))],
            deltas=[[91, 92], [93, 94]],
            outputs=[[101, 102, 103], [201, 202, 203]],
            expected_output_tokens=3,
            new_input_tokens=[10, 10],
            wall_s=2.0,
            ttft_s=[0.1, 0.3],
            e2e_s=[0.4, 0.8],
            cached_tokens=[0, 8],
            errors=[None, None],
        )

        self.assertEqual(row["success_count"], 2)
        self.assertEqual(row["error_count"], 0)
        self.assertEqual(row["output_tokens"], 6)
        self.assertEqual(row["successful_new_input_tokens"], 20)
        self.assertEqual(row["useful_new_tokens"], 26)
        self.assertEqual(row["output_tok_s"], 3.0)
        self.assertEqual(row["useful_new_token_tok_s"], 13.0)
        self.assertEqual(row["p50_ttft_s"], 0.2)
        self.assertEqual(row["p95_ttft_s"], 0.29)
        self.assertEqual(row["p50_e2e_latency_s"], 0.6)
        self.assertEqual(row["p95_e2e_latency_s"], 0.78)
        self.assertEqual(row["cached_tokens_total"], 8)
        self.assertEqual(row["p50_cached_tokens"], 4.0)
        self.assertEqual(row["p95_cached_tokens"], 7.6)
        self.assertEqual(len(row["prompt_token_ids_sha256"]), 64)
        self.assertEqual(len(row["delta_token_ids_sha256"]), 64)
        self.assertEqual(len(row["request_output_token_ids_sha256"]), 64)
        self.assertTrue(row["output_fingerprint_complete"])

    def test_summarize_turn_counts_only_successful_useful_tokens(self) -> None:
        row = summarize_turn(
            turn_index=1,
            session_ids=["a", "b"],
            prompts=[[1, 2, 3], [4, 5, 6]],
            deltas=[[2], [5]],
            outputs=[[7, 8], [9]],
            expected_output_tokens=2,
            new_input_tokens=[1, 1],
            wall_s=1.0,
        )

        self.assertEqual(row["success_count"], 1)
        self.assertEqual(row["error_count"], 1)
        self.assertEqual(row["output_tokens"], 2)
        self.assertEqual(row["successful_new_input_tokens"], 1)
        self.assertEqual(row["useful_new_token_tok_s"], 3.0)
        self.assertFalse(row["output_fingerprint_complete"])
        self.assertIn("expected 2 output tokens", row["errors"][0]["error"])

    def test_percentile_and_run_summary_use_all_available_request_latencies(self) -> None:
        self.assertEqual(percentile([1.0, 3.0], 0.50), 2.0)
        self.assertAlmostEqual(percentile([1.0, 3.0], 0.95), 2.9)
        turn = summarize_turn(
            turn_index=0,
            session_ids=["a", "b"],
            prompts=[[1], [2]],
            deltas=[[3], [4]],
            outputs=[[5], [6]],
            expected_output_tokens=1,
            new_input_tokens=[1, 1],
            wall_s=2.0,
            ttft_s=[0.1, 0.3],
            e2e_s=[0.5, 0.9],
            cached_tokens=[0, 4],
        )

        summary = summarize_run([turn], requested_turns=1)

        self.assertTrue(summary["all_turns_recorded"])
        self.assertEqual(summary["output_tok_s"], 1.0)
        self.assertEqual(summary["useful_new_token_tok_s"], 2.0)
        self.assertEqual(summary["p50_ttft_s"], 0.2)
        self.assertEqual(summary["p95_e2e_latency_s"], 0.88)
        self.assertEqual(summary["cached_tokens_total"], 4)
        self.assertEqual(summary["completed_requests_per_s"], 1.0)
        self.assertTrue(summary["cache_telemetry_complete"])
        self.assertEqual(summary["turn_0"]["output_tok_s"], 1.0)
        self.assertEqual(summary["continuation_turns"]["turn_rows"], 0)

    def test_run_summary_separates_turn_zero_and_continuations(self) -> None:
        rows = [
            summarize_turn(
                turn_index=turn_index,
                session_ids=["a", "b"],
                prompts=[[1], [2]],
                deltas=[[], []] if turn_index == 0 else [[3], [4]],
                outputs=[[5, 6], [7, 8]],
                expected_output_tokens=2,
                new_input_tokens=[1, 1],
                wall_s=2.0 if turn_index == 0 else 1.0,
                cached_tokens=[0, 0] if turn_index == 0 else [1, 1],
            )
            for turn_index in range(3)
        ]

        summary = summarize_run(rows, requested_turns=3)

        self.assertEqual(summary["output_tok_s"], 3.0)
        self.assertEqual(summary["turn_0"]["output_tok_s"], 2.0)
        self.assertEqual(summary["continuation_turns"]["output_tok_s"], 4.0)
        self.assertEqual(
            summary["continuation_turns"]["completed_requests_per_s"],
            2.0,
        )
        self.assertEqual(summary["cached_tokens_available_count"], 6)

    def test_cached_token_extractors_use_actual_incumbent_fields(self) -> None:
        vllm_outputs = [
            types.SimpleNamespace(num_cached_tokens=7, metrics=None),
            types.SimpleNamespace(
                num_cached_tokens=None,
                metrics=types.SimpleNamespace(num_cached_tokens=11),
            ),
            types.SimpleNamespace(
                num_cached_tokens=None,
                metrics=types.SimpleNamespace(cached_tokens=13),
            ),
            types.SimpleNamespace(num_cached_tokens=None, metrics=None),
        ]
        sglang_outputs = [
            {"meta_info": {"cached_tokens": 5}},
            {"meta_info": {"cached_tokens": 0}},
            {"meta_info": {}},
        ]

        self.assertEqual(
            extract_vllm_cached_tokens(vllm_outputs),
            [7, 11, 13, None],
        )
        self.assertEqual(
            extract_sglang_cached_tokens(sglang_outputs),
            [5, 0, None],
        )


if __name__ == "__main__":
    unittest.main()
