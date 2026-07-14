import json
import re
import threading
import types
import unittest

from experiments.open_webui_multiturn_bench import (
    BenchmarkConfig,
    PendingRequest,
    RequestPlan,
    SocketEventTracker,
    build_request_plan,
    build_workload,
    exact_initial_content,
    redact_command,
    redact_secrets,
    render_chat_token_ids,
    request_order_indices,
    run_benchmark,
    summarize_records,
    validate_records,
    validate_wkvm_session_metrics,
)


class _FakeTokenizer:
    def __init__(self) -> None:
        self.return_dict_values = []

    @staticmethod
    def _words(text):
        return re.findall(r"\S+", text)

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("benchmark text encoding must disable special tokens")
        return [100 + index for index, _word in enumerate(self._words(text))]

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_dict,
    ):
        self.return_dict_values.append(return_dict)
        if not tokenize:
            raise AssertionError("benchmark must request token IDs")
        token_ids = [1]
        for message in messages:
            token_ids.append(10 if message["role"] == "user" else 11)
            token_ids.extend(self.encode(message["content"], add_special_tokens=False))
            token_ids.append(12)
        if add_generation_prompt:
            token_ids.append(11)
        return token_ids


class _FakeOpenWebUITransport:
    def __init__(self, tokenizer, model, output_tokens):
        self.tokenizer = tokenizer
        self.model = model
        self.output_tokens = output_tokens
        self.session_id = "socket-test"
        self.event_handler = None
        self.payloads = []
        self.closed = False
        self._lock = threading.Lock()
        self._next_chat = 0

    def connect(self, event_handler):
        self.event_handler = event_handler

    def close(self):
        self.closed = True

    def get_version(self):
        return {"version": "0.10.2", "deployment_id": "fake"}

    def get_models(self):
        return {"data": [{"id": self.model, "name": self.model}]}

    def post_completion(self, payload):
        with self._lock:
            self.payloads.append(json.loads(json.dumps(payload)))
            chat_id = payload.get("chat_id")
            if chat_id is None:
                chat_id = f"chat-{self._next_chat:04d}"
                self._next_chat += 1
        message_id = payload["message_ids"][0]["message_id"]
        prompt_tokens = len(
            render_chat_token_ids(
                self.tokenizer,
                payload["messages"],
                add_generation_prompt=True,
            )
        )
        turn_index = (len(payload["messages"]) - 1) // 2
        output_text = " ".join(
            f"answer{turn_index}_{index}" for index in range(self.output_tokens)
        )
        partial_text = output_text.split()[0]
        partial = {
            "chat_id": chat_id,
            "message_id": message_id,
            "data": {
                "type": "chat:completion",
                "data": {
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": partial_text}
                            ],
                        }
                    ]
                },
            },
        }
        final = {
            "chat_id": chat_id,
            "message_id": message_id,
            "data": {
                "type": "chat:completion",
                "data": {
                    "done": True,
                    "output": [
                        {
                            "type": "message",
                            "content": [
                                {"type": "output_text", "text": output_text}
                            ],
                        }
                    ],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": self.output_tokens,
                        "total_tokens": prompt_tokens + self.output_tokens,
                        "prompt_tokens_details": {
                            "cached_tokens": 0 if turn_index == 0 else 2
                        },
                    },
                },
            },
        }
        self.event_handler(partial)
        self.event_handler(final)
        return {
            "status": True,
            "task_ids": [f"task-{message_id}"],
            "chat_id": chat_id,
        }


class TestOpenWebUIMultiturnBench(unittest.TestCase):
    def test_exact_text_workload_is_deterministic_and_requests_plain_token_ids(self):
        tokenizer = _FakeTokenizer()
        first = build_workload(
            tokenizer,
            sessions=2,
            turns=3,
            initial_context_tokens=24,
            turn_input_tokens=5,
        )
        second = build_workload(
            tokenizer,
            sessions=2,
            turns=3,
            initial_context_tokens=24,
            turn_input_tokens=5,
        )

        self.assertEqual(first, second)
        self.assertEqual(first.initial_rendered_token_counts, [24, 24])
        self.assertEqual(first.turn_content_token_counts, [[5, 5], [5, 5]])
        self.assertEqual(len(set(first.initial_contents)), 2)
        self.assertTrue(tokenizer.return_dict_values)
        self.assertEqual(set(tokenizer.return_dict_values), {False})

        content, count = exact_initial_content(
            tokenizer,
            session_index=9,
            target_rendered_tokens=24,
        )
        self.assertTrue(content.startswith("SESSION-0009-"))
        self.assertEqual(count, 24)

    def test_frontend_payload_has_usage_fixed_sampling_and_no_background_work(self):
        tokenizer = _FakeTokenizer()
        config = BenchmarkConfig(
            open_webui_url="http://example.test",
            model="gemma-test",
            run_id="payload-test",
            sessions=1,
            turns=1,
            initial_context_tokens=24,
            turn_input_tokens=5,
            output_tokens_per_turn=3,
        )
        workload = build_workload(
            tokenizer,
            sessions=1,
            turns=1,
            initial_context_tokens=24,
            turn_input_tokens=5,
        )
        conversation = types.SimpleNamespace(
            logical_session_id="session-0000",
            messages=[],
            chat_id=None,
            last_assistant_message_id=None,
        )

        plan = build_request_plan(
            config,
            tokenizer,
            conversation,
            session_index=0,
            turn_index=0,
            request_order_index=0,
            user_content=workload.initial_contents[0],
            socket_session_id="socket-1",
        )

        self.assertTrue(plan.payload["stream"])
        self.assertEqual(plan.payload["stream_options"], {"include_usage": True})
        self.assertEqual(plan.payload["background_tasks"], {})
        self.assertEqual(plan.payload["features"], {})
        self.assertEqual(plan.payload["params"]["temperature"], 0)
        self.assertEqual(plan.payload["params"]["function_calling"], "legacy")
        self.assertEqual(plan.payload["params"]["max_tokens"], 3)
        self.assertEqual(
            plan.payload["params"]["custom_params"],
            {"ignore_eos": True},
        )
        self.assertEqual(plan.local_prompt_tokens, 24)
        self.assertEqual(plan.unique_logical_input_tokens, 24)
        self.assertNotIn("chat_id", plan.payload)

    def test_tracker_uses_first_nonempty_cumulative_output_and_survives_pre_ack_done(self):
        tracker = SocketEventTracker()
        plan = RequestPlan(
            logical_session_id="session-0000",
            session_index=0,
            turn_index=0,
            request_order_index=0,
            message_id="message-1",
            user_message_id="user-1",
            expected_chat_id=None,
            payload={},
            local_prompt_tokens=10,
            unique_logical_input_tokens=10,
            prompt_token_ids_sha256="a" * 64,
            user_content_sha256="b" * 64,
            payload_sha256="c" * 64,
        )
        pending = PendingRequest(plan=plan, request_start_ns=1)
        tracker.register(pending)

        tracker.handle_event(
            {
                "chat_id": "chat-1",
                "message_id": "message-1",
                "data": {
                    "type": "chat:completion",
                    "data": {"usage": {"prompt_tokens": 10}},
                },
            }
        )
        self.assertIsNone(pending.first_content_ns)
        tracker.handle_event(
            {
                "chat_id": "chat-1",
                "message_id": "message-1",
                "data": {
                    "type": "chat:completion",
                    "data": {
                        "done": True,
                        "output": [
                            {
                                "type": "message",
                                "content": [
                                    {"type": "output_text", "text": "one two"}
                                ],
                            }
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 2,
                        },
                    },
                },
            }
        )
        self.assertTrue(pending.done)
        self.assertEqual(pending.output_text, "one two")
        tracker.record_ack(
            "message-1",
            {"status": True, "chat_id": "chat-1", "task_ids": ["task-1"]},
            pending.terminal_ns + 1,
        )
        self.assertIsNone(pending.error)
        self.assertEqual(pending.ack_chat_id, "chat-1")

    def test_fake_b2x2_exercises_ack_socket_history_and_all_three_rates(self):
        tokenizer = _FakeTokenizer()
        config = BenchmarkConfig(
            open_webui_url="http://example.test",
            model="gemma-test",
            run_id="fake-b2x2",
            sessions=2,
            turns=2,
            initial_context_tokens=24,
            turn_input_tokens=5,
            output_tokens_per_turn=3,
            request_order_policy="alternating",
        )
        transport = _FakeOpenWebUITransport(tokenizer, config.model, 3)

        artifact = run_benchmark(config, tokenizer, transport)

        self.assertEqual(artifact["status"], "passed")
        self.assertTrue(artifact["validation"]["passed"])
        self.assertTrue(transport.closed)
        self.assertEqual(len(artifact["requests"]), 4)
        self.assertEqual(artifact["summary"]["success_count"], 4)
        self.assertEqual(artifact["summary"]["generated_output_tokens"], 12)
        self.assertEqual(
            artifact["summary"]["unique_logical_input_tokens"],
            58,
        )
        self.assertEqual(artifact["summary"]["total_application_tokens"], 70)
        expected_api_total = sum(
            row["provider_prompt_tokens"] + row["provider_completion_tokens"]
            for row in artifact["requests"]
        )
        self.assertEqual(
            artifact["summary"]["api_accounted_total_tokens"],
            expected_api_total,
        )
        wall = artifact["summary"]["synchronized_wall_s"]
        self.assertAlmostEqual(
            artifact["summary"]["e2e_generated_output_tok_s"] * wall,
            12,
            places=3,
        )
        self.assertAlmostEqual(
            artifact["summary"]["api_accounted_total_tok_s"] * wall,
            expected_api_total,
            places=3,
        )
        self.assertAlmostEqual(
            artifact["summary"]["total_application_goodput_tok_s"] * wall,
            70,
            places=3,
        )
        self.assertEqual(artifact["session_cache"]["stable_chat_id_sessions"], 2)
        self.assertEqual(artifact["open_webui"]["logical_conversations"], 2)
        self.assertEqual(
            artifact["open_webui"]["client_layout"],
            "one_authenticated_socket_multiple_logical_conversations",
        )
        self.assertEqual(
            len({row["chat_id_sha256"] for row in artifact["requests"]}),
            2,
        )
        self.assertEqual(artifact["turns"][1]["request_order"], [1, 0])
        continuation_payloads = [
            payload for payload in transport.payloads if payload.get("chat_id")
        ]
        self.assertEqual(len(continuation_payloads), 2)
        self.assertTrue(all(len(payload["messages"]) == 3 for payload in continuation_payloads))

    def test_summary_with_missing_usage_refuses_all_token_rates(self):
        records = [
            {
                "request_start_ns": 0,
                "terminal_ns": 1_000_000_000,
                "transport_success": True,
                "accounting_valid": False,
                "provider_prompt_tokens": None,
                "provider_completion_tokens": None,
                "unique_logical_input_tokens": 10,
                "ack_latency_s": 0.1,
                "ui_path_ttft_s": 0.2,
                "e2e_latency_s": 1.0,
            }
        ]

        summary = summarize_records(records, expected_requests=1)

        self.assertFalse(summary["accounting_complete"])
        self.assertIsNone(summary["e2e_generated_output_tok_s"])
        self.assertIsNone(summary["api_accounted_total_tok_s"])
        self.assertIsNone(summary["total_application_goodput_tok_s"])
        self.assertEqual(summary["completed_requests_per_s"], 1.0)

    def test_validation_rejects_missing_usage_and_unstable_chat(self):
        config = BenchmarkConfig(
            open_webui_url="http://example.test",
            model="gemma-test",
            run_id="invalid",
            sessions=1,
            turns=1,
            initial_context_tokens=24,
            turn_input_tokens=5,
            output_tokens_per_turn=3,
        )
        record = {
            "logical_session_id": "session-0000",
            "turn_index": 0,
            "transport_success": True,
            "usage_complete": False,
            "provider_completion_tokens": None,
            "provider_prompt_tokens": None,
            "local_rendered_prompt_tokens": 24,
            "chat_id_stable": False,
            "chat_id_sha256": None,
            "output_text_chars": 0,
            "ui_path_ttft_s": None,
            "unique_logical_input_tokens": 24,
        }

        validation = validate_records([record], config)

        self.assertFalse(validation["passed"])
        self.assertGreaterEqual(validation["issue_count"], 5)

    def test_wkvm_session_metrics_require_residency_and_every_reuse(self):
        config = BenchmarkConfig(
            open_webui_url="http://example.test",
            model="gemma-test",
            run_id="metrics",
            sessions=2,
            turns=3,
            provider_metrics_url="http://provider.test/metrics",
            require_wkvm_session_reuse=True,
        )
        snapshot = {
            "phase": "after-run",
            "data": {
                "server": {"chat_sessions": 2},
                "engine": {
                    "parked_sessions": 2,
                    "resident_sessions": 2,
                    "session_reuse_hits": 4,
                    "session_reuse_misses": 0,
                    "full_reprefill_turns": 0,
                },
            },
        }

        self.assertEqual(validate_wkvm_session_metrics([snapshot], config), [])
        snapshot["data"]["engine"]["session_reuse_hits"] = 3
        issues = validate_wkvm_session_metrics([snapshot], config)
        self.assertEqual(len(issues), 1)
        self.assertIn("expected 4, got 3", issues[0])

    def test_secret_redaction_is_recursive(self):
        redacted = redact_secrets(
            {
                "Authorization": "Bearer secret",
                "nested": {
                    "api_key": "key",
                    "safe": "value",
                    "items": [{"password": "pw"}],
                },
            }
        )

        self.assertEqual(redacted["Authorization"], "<redacted>")
        self.assertEqual(redacted["nested"]["api_key"], "<redacted>")
        self.assertEqual(redacted["nested"]["safe"], "value")
        self.assertEqual(
            redact_secrets({"prompt_tokens": 12, "completion_tokens": 3}),
            {"prompt_tokens": 12, "completion_tokens": 3},
        )
        self.assertEqual(
            redacted["nested"]["items"][0]["password"],
            "<redacted>",
        )
        self.assertEqual(
            redact_command("serve --api-key topsecret --max-tokens 128"),
            "serve --api-key '<redacted>' --max-tokens 128",
        )
        self.assertEqual(
            redact_command("OPENAI_API_KEY=topsecret serve"),
            "'OPENAI_API_KEY=<redacted>' serve",
        )

    def test_request_order_matches_existing_multiturn_contract(self):
        self.assertEqual(request_order_indices(4, 0, "alternating"), [0, 1, 2, 3])
        self.assertEqual(request_order_indices(4, 1, "alternating"), [3, 2, 1, 0])
        shuffled = request_order_indices(8, 3, "seeded-shuffle", seed=7)
        self.assertEqual(sorted(shuffled), list(range(8)))
        self.assertEqual(
            shuffled,
            request_order_indices(8, 3, "seeded-shuffle", seed=7),
        )


if __name__ == "__main__":
    unittest.main()
