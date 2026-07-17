from __future__ import annotations

import hashlib
import json
import re
import unittest

from experiments.open_webui_demo_scenario import (
    CAVEATS,
    build_parser,
    build_report,
    build_scenario,
    canonical_sha256,
    encode_text,
    percentile,
    report_markdown,
)


class _FakeTokenizer:
    name_or_path = "/private/models/fake-gemma"
    vocab_size = 32_000
    chat_template = "<BOS> <USER> {{ content }} <END_USER> <ASSISTANT>"
    bos_token_id = 1
    eos_token_id = 2

    @staticmethod
    def _matches(text: str):
        return list(re.finditer(r"\S+", text))

    @staticmethod
    def _token_id(word: str) -> int:
        return 100 + int(hashlib.sha256(word.encode()).hexdigest()[:8], 16)

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("response counting must omit special tokens")
        return [self._token_id(match.group()) for match in self._matches(text)]

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        if add_special_tokens or not return_offsets_mapping:
            raise AssertionError("unexpected tokenizer call")
        matches = self._matches(text)
        return {
            "input_ids": [self._token_id(match.group()) for match in matches],
            "offset_mapping": [match.span() for match in matches],
        }

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_dict=False,
    ):
        if len(messages) != 1 or messages[0]["role"] != "user":
            raise AssertionError("fake tokenizer only supports one user message")
        rendered = f"<BOS> <USER> {messages[0]['content']} <END_USER>"
        if add_generation_prompt:
            rendered += " <ASSISTANT>"
        if not tokenize:
            return rendered
        token_ids = self.encode(rendered, add_special_tokens=False)
        return {"input_ids": token_ids} if return_dict else token_ids


def _capture_for(scenario):
    answers = {
        "reasoning": (
            "The equations give x + (x + USD 1.00) = USD 1.10. "
            "FINAL: USD 0.05"
        ),
        "code": (
            "```python\ndef is_palindrome(text):\n"
            "    normalized = ''.join(c.lower() for c in text if c.isalnum())\n"
            "    return normalized == normalized[::-1]\n```"
        ),
        "json": '{"engine":"WKVM","slots":4,"status":"ready"}',
        "systems": (
            "A bounded queue applies backpressure before overload can exhaust "
            "the server's resources."
        ),
    }
    sessions = []
    for index, prompt in enumerate(scenario["concurrent_prompts"]):
        sessions.append(
            {
                "prompt_id": prompt["prompt_id"],
                "label": prompt["label"],
                "first_turn": {
                    "timing": {
                        "ttft_s": float(index + 1),
                        "e2e_s": float(index + 4),
                    },
                    "response_text": answers[prompt["prompt_id"]],
                    "error": None,
                },
                "follow_up": {
                    "prompt_id": "common-follow-up",
                    "label": "Common Follow-up",
                    "timing": {
                        "ttft_s": 0.25 + index * 0.1,
                        "e2e_s": 0.75 + index * 0.1,
                    },
                    "response_text": "The previous response states its core result.",
                    "error": None,
                },
            }
        )
    return {
        "schema_version": 1,
        "kind": "wkvm.open_webui.live_capture",
        "captured_at": "2026-07-17T01:02:03Z",
        "scenario": {"sha256": scenario["scenario_sha256"]},
        "provenance": {"browser": {"channel": "chromium", "version": "test"}},
        "acts": {
            "long_prompt": {
                "prompt_id": "long-context-needle",
                "label": "Long Context Needle",
                "timing": {"ttft_s": 5.0, "e2e_s": 10.0},
                "response_text": (
                    "The codename is BLUE-742, the city is Samarkand, and the "
                    "checksum is lantern."
                ),
                "error": None,
            },
            "concurrency": {
                "count": 4,
                "synchronized_countdown_s": 3,
                "sessions": sessions,
            },
        },
        "errors": [],
    }


def _capture_with_telemetry(scenario):
    capture = _capture_for(scenario)
    capture["summary"] = {
        "turns_attempted": 9,
        "turns_succeeded": 9,
        "turns_failed": 0,
        "capture_errors": 0,
        "probe_errors": 0,
        "success": True,
    }
    capture["acts"]["long_prompt"].update(
        {
            "gpu": {
                "sample_count": 37,
                "devices": [
                    {
                        "index": 0,
                        "name": "NVIDIA GeForce RTX 4090",
                        "total_mib": 24_564,
                        "baseline_used_mib": 17_370,
                        "peak_used_mib": 17_770,
                    }
                ],
                "error": None,
            },
            "provider": {
                "before": {
                    "metrics": {
                        "values": {
                            "engine": {"max_running": 0, "max_runnable_rows": 0}
                        }
                    }
                },
                "after": {
                    "metrics": {
                        "values": {
                            "engine": {
                                "max_running": 1,
                                "max_runnable_rows": 1,
                                "persistent_padded_decode": True,
                                "persistent_padded_decode_steps": 128,
                                "persistent_padded_decode_cuda_graph": False,
                                "use_native_gemma_forward": True,
                                "native_gemma_attention_backend": "sdpa_single_gqa",
                                "native_gemma_projection_backend": "separate",
                                "native_gemma_weight_backend": "hf_live",
                                "native_gemma_checkpoint_loader": True,
                            }
                        }
                    }
                },
                "delta": {
                    "server": {
                        "total_requests": 1,
                        "total_errors": 0,
                        "total_cancelled": 0,
                        "total_timed_out": 0,
                    }
                },
            },
        }
    )
    capture["acts"]["concurrency"].update(
        {
            "gpu": {
                "sample_count": 76,
                "devices": [
                    {
                        "index": 0,
                        "name": "NVIDIA GeForce RTX 4090",
                        "total_mib": 24_564,
                        "baseline_used_mib": 17_770,
                        "peak_used_mib": 17_810,
                    }
                ],
                "error": None,
            },
            "provider": {
                "before": {
                    "metrics": {
                        "values": {
                            "engine": {"max_running": 1, "max_runnable_rows": 1}
                        }
                    }
                },
                "after_first_turn": {
                    "metrics": {
                        "values": {
                            "engine": {"max_running": 4, "max_runnable_rows": 4}
                        }
                    }
                },
                "after": {
                    "metrics": {
                        "values": {
                            "engine": {
                                "max_running": 4,
                                "max_runnable_rows": 4,
                                "persistent_padded_decode": True,
                                "persistent_padded_decode_steps": 128,
                                "persistent_padded_decode_cuda_graph": False,
                                "use_native_gemma_forward": True,
                                "native_gemma_attention_backend": "sdpa_single_gqa",
                                "native_gemma_projection_backend": "separate",
                                "native_gemma_weight_backend": "hf_live",
                                "native_gemma_checkpoint_loader": True,
                            }
                        }
                    }
                },
                "first_turn_delta": {
                    "server": {
                        "total_requests": 4,
                        "total_errors": 0,
                        "total_cancelled": 0,
                        "total_timed_out": 0,
                    }
                },
                "follow_up_delta": {
                    "server": {
                        "total_requests": 4,
                        "total_errors": 0,
                        "total_cancelled": 0,
                        "total_timed_out": 0,
                    },
                    "engine": {
                        "sessions_opened": 0,
                        "session_reuse_hits": 4,
                        "prefix_tokens_reused": 333,
                    },
                },
                "delta": {
                    "server": {
                        "total_requests": 8,
                        "total_errors": 0,
                        "total_cancelled": 0,
                        "total_timed_out": 0,
                    }
                },
                "follow_up_session_reuse_delta": {
                    "session_reuse_hits": 4,
                    "prefix_tokens_reused": 333,
                },
            },
        }
    )
    return capture


class TestOpenWebUIDemoScenario(unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.tokenizer = _FakeTokenizer()
        self.scenario = build_scenario(self.tokenizer)

    def test_default_long_prompt_is_exact_and_hashes_are_deterministic(self):
        second = build_scenario(self.tokenizer)

        self.assertEqual(self.scenario, second)
        self.assertEqual(self.scenario["long_prompt"]["rendered_token_count"], 12_000)
        self.assertEqual(
            self.scenario["scenario_sha256"],
            canonical_sha256(
                {
                    key: value
                    for key, value in self.scenario.items()
                    if key != "scenario_sha256"
                }
            ),
        )
        needle = self.scenario["long_prompt"]["needle"]
        self.assertIsNotNone(needle["rendered_token_index"])
        self.assertLessEqual(abs(needle["rendered_token_index"] - 256), 8)
        self.assertEqual(needle["position_method"], "fast_tokenizer_offset_mapping")
        self.assertEqual(
            needle["facts"],
            {
                "codename": "BLUE-742",
                "city": "Samarkand",
                "checksum": "lantern",
            },
        )

        prompt_hashes = [
            self.scenario["long_prompt"]["prompt_sha256"],
            *[
                prompt["prompt_sha256"]
                for prompt in self.scenario["concurrent_prompts"]
            ],
        ]
        self.assertEqual(len(prompt_hashes), len(set(prompt_hashes)))
        self.assertTrue(all(re.fullmatch(r"[0-9a-f]{64}", value) for value in prompt_hashes))
        self.assertNotIn("/private/models", json.dumps(self.scenario))
        self.assertIsNone(self.scenario["tokenizer"]["identity"])

    def test_report_validates_outputs_counts_tokens_and_calculates_percentiles(self):
        capture = _capture_for(self.scenario)

        report = build_report(capture, self.scenario, self.tokenizer)

        self.assertEqual(report["status"], "passed")
        self.assertTrue(report["summary"]["overall_passed"])
        self.assertEqual(report["summary"]["offered_concurrency"], 4)
        self.assertEqual(report["summary"]["request_count"], 4)
        self.assertEqual(report["summary"]["success_count"], 4)
        self.assertEqual(report["summary"]["all_request_count"], 9)
        self.assertEqual(report["summary"]["all_success_count"], 9)
        self.assertEqual(report["summary"]["all_error_count"], 0)
        self.assertAlmostEqual(report["summary"]["ttft_p50_s"], 2.5)
        self.assertAlmostEqual(report["summary"]["ttft_p95_s"], 3.85)
        self.assertAlmostEqual(report["summary"]["e2e_p50_s"], 5.5)
        self.assertAlmostEqual(report["summary"]["e2e_p95_s"], 6.85)
        expected_tokens = sum(
            len(encode_text(self.tokenizer, record["response_text"]))
            for record in report["records"]
        )
        self.assertEqual(report["summary"]["total_output_tokens"], expected_tokens)
        self.assertTrue(all(record["validation"]["passed"] for record in report["records"]))
        self.assertTrue(report["provenance"]["tokenizer"]["match"]["matched"])
        self.assertTrue(report["provenance"]["scenario"]["capture_match"])
        self.assertEqual(
            report["provenance"]["capture"]["browser"]["version"], "test"
        )
        self.assertAlmostEqual(percentile([1, 2, 3, 4], 0.95), 3.85)

    def test_invalid_classic_json_fails_validation_and_is_reported(self):
        capture = _capture_for(self.scenario)
        json_session = next(
            session
            for session in capture["acts"]["concurrency"]["sessions"]
            if session["prompt_id"] == "json"
        )
        json_session["first_turn"]["response_text"] = (
            '```json\n{"engine":"WKVM","slots":4,"status":"ready"}\n```'
        )

        report = build_report(capture, self.scenario, self.tokenizer)

        self.assertEqual(report["status"], "failed")
        failed = [record for record in report["records"] if not record["success"]]
        self.assertEqual([record["prompt_id"] for record in failed], ["json"])
        self.assertIn("response is not strict JSON", "\n".join(report["errors"]))

    def test_markdown_states_scope_and_claim_caveats(self):
        report = build_report(
            _capture_with_telemetry(self.scenario),
            self.scenario,
            self.tokenizer,
        )

        markdown = report_markdown(report)

        self.assertIn("**Offered UI concurrency:** 4 chats", markdown)
        self.assertIn("Classic first turn", markdown)
        self.assertIn(
            "| Long context | 17,370 MiB | 17,770 MiB | 1 | 0 | 0 | 0 | 1 | 1 |",
            markdown,
        )
        self.assertIn(
            "| Concurrency | 17,770 MiB | 17,810 MiB | 8 | 0 | 0 | 0 | 4 | 4 |",
            markdown,
        )
        self.assertIn("| Classic first turn | 4 | 0 | 0 | 0 |", markdown)
        self.assertIn("| Common follow-up | 4 | 0 | 0 | 0 |", markdown)
        self.assertIn(
            "**Follow-up reuse:** 4 reuse hits; 0 sessions opened; "
            "333 prefix tokens reused.",
            markdown,
        )
        self.assertIn("**Capture health:** 0 capture errors; 0 probe errors.", markdown)
        self.assertIn("`scenario.claim_scope.semantics`", markdown)
        self.assertIn("`persistent_padded_decode_cuda_graph=false`", markdown)
        self.assertIn("`native_gemma_attention_backend=sdpa_single_gqa`", markdown)
        for caveat in CAVEATS:
            self.assertIn(caveat, markdown)
        self.assertIn("routed_span_approximate", markdown)
        self.assertIn("not a vLLM/SGLang comparison", markdown)
        self.assertIn("not a controlled load test", markdown)

    def test_report_preserves_optional_runtime_telemetry(self):
        report = build_report(
            _capture_with_telemetry(self.scenario),
            self.scenario,
            self.tokenizer,
        )

        telemetry = report["telemetry"]
        self.assertEqual(
            telemetry["long_prompt"]["gpu"]["whole_gpu_baseline_used_mib"],
            17_370,
        )
        self.assertEqual(
            telemetry["long_prompt"]["gpu"]["whole_gpu_peak_used_mib"],
            17_770,
        )
        self.assertEqual(
            telemetry["concurrency"]["provider"]["high_water"],
            {"max_running": 4, "max_runnable_rows": 4},
        )
        self.assertEqual(
            telemetry["concurrency"]["provider"]["request_counts"],
            {
                "total_requests": 8,
                "total_errors": 0,
                "total_cancelled": 0,
                "total_timed_out": 0,
            },
        )
        self.assertEqual(
            telemetry["concurrency"]["provider"]["follow_up_reuse"],
            {
                "session_reuse_hits": 4,
                "sessions_opened": 0,
                "prefix_tokens_reused": 333,
            },
        )
        self.assertEqual(
            telemetry["capture"], {"capture_errors": 0, "probe_errors": 0}
        )
        self.assertEqual(
            telemetry["launch"]["semantics"],
            {
                "value": "routed_span_approximate",
                "source": "scenario.claim_scope.semantics",
            },
        )
        engine_config = telemetry["launch"]["provider_engine_config"]
        self.assertEqual(
            engine_config["source"],
            "capture.acts.concurrency.provider.after.metrics.values.engine",
        )
        self.assertFalse(
            engine_config["values"]["persistent_padded_decode_cuda_graph"]
        )
        self.assertEqual(report["summary"]["capture_error_count"], 0)
        self.assertEqual(report["summary"]["probe_error_count"], 0)

    def test_missing_runtime_telemetry_is_reported_as_unavailable(self):
        report = build_report(_capture_for(self.scenario), self.scenario, self.tokenizer)

        self.assertEqual(report["status"], "passed")
        self.assertIsNone(
            report["telemetry"]["long_prompt"]["gpu"][
                "whole_gpu_baseline_used_mib"
            ]
        )
        self.assertIsNone(
            report["telemetry"]["concurrency"]["provider"]["high_water"][
                "max_running"
            ]
        )
        self.assertIsNone(report["summary"]["capture_error_count"])
        self.assertIsNone(report["summary"]["probe_error_count"])
        self.assertIn("| Long context | n/a | n/a | n/a", report_markdown(report))

    def test_cli_shape_keeps_12000_default_and_required_outputs(self):
        parser = build_parser()

        build_args = parser.parse_args(
            ["build", "--tokenizer-path", "public/model", "--json", "scenario.json"]
        )
        report_args = parser.parse_args(
            [
                "report",
                "capture.json",
                "--scenario",
                "scenario.json",
                "--json",
                "report.json",
                "--markdown",
                "report.md",
            ]
        )

        self.assertEqual(build_args.long_rendered_tokens, 12_000)
        self.assertEqual(str(report_args.capture_json), "capture.json")
        self.assertEqual(str(report_args.markdown), "report.md")


if __name__ == "__main__":
    unittest.main()
