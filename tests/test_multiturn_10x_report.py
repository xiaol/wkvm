import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.bench_prompt_utils import generated_output_fingerprint
from experiments.multiturn_10x_report import (
    _artifact_record,
    build_parser,
    build_report,
    render_markdown,
)


CAMPAIGN_ID = "publication-campaign-test"
GIT_COMMIT = "f" * 40
MEMORY_CEILING_MIB = 24_000
SESSIONS = 16
TURNS = 8
CONTEXT = 36_864
DELTA = 32
OUTPUT = 64
REQUIRED_MODEL_LEN = 37_600


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def stable_sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def workload_fingerprints() -> dict:
    return {
        "initial_prompts": {
            "schema": "wkvm.prompt_token_ids.sha256.v1",
            "prompt_token_source": "synthetic_lcg",
            "prompt_count": SESSIONS,
            "prompt_total_tokens": SESSIONS * CONTEXT,
            "prompt_lengths": [CONTEXT] * SESSIONS,
            "prompt_token_ids_sha256": stable_sha256("initial-prompts"),
        },
        "turn_deltas": [
            {
                "schema": "wkvm.prompt_token_ids.sha256.v1",
                "prompt_token_source": "synthetic_lcg",
                "prompt_count": SESSIONS,
                "prompt_total_tokens": SESSIONS * DELTA,
                "prompt_lengths": [DELTA] * SESSIONS,
                "prompt_token_ids_sha256": stable_sha256(f"delta-{turn}"),
            }
            for turn in range(1, TURNS)
        ],
    }


def make_trace(repeat_id: str, source_run_id: str) -> dict:
    repeat_number = int(repeat_id.removeprefix("r"))
    turn_outputs = [
        [
            [
                (repeat_number * 10_000 + turn * 1_000 + session * 100 + token)
                % 262_144
                for token in range(OUTPUT)
            ]
            for session in range(SESSIONS)
        ]
        for turn in range(TURNS)
    ]
    request_ids = [f"session-{index:04d}" for index in range(SESSIONS)]
    output_fingerprints = [
        generated_output_fingerprint(
            zip(request_ids, outputs, strict=True)
        )
        for outputs in turn_outputs
    ]
    trace_workload = {
        "sessions": SESSIONS,
        "turns": TURNS,
        "output_tokens_per_turn": OUTPUT,
        "prompt_token_source": "synthetic_lcg",
        "fingerprints": workload_fingerprints(),
    }
    contract = {
        "schema": "wkvm.gemma_shared_history_trace.v1",
        "workload": trace_workload,
        "turn_outputs": turn_outputs,
    }
    return {
        **contract,
        "trace_sha256": canonical_sha256(contract),
        "output_fingerprints": output_fingerprints,
        "source": {
            "benchmark_identity": {
                "campaign_id": CAMPAIGN_ID,
                "repeat_id": repeat_id,
                "run_id": source_run_id,
            }
        },
    }


def write_trace(directory: Path, repeat_id: str, source_run_id: str) -> tuple[Path, dict]:
    trace = make_trace(repeat_id, source_run_id)
    path = directory / f"{repeat_id}.trace.json"
    path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
    return path, trace


def make_artifact(
    engine: str,
    rate: float,
    *,
    repeat_id: str = "r1",
    trace_path: Path | None = None,
    trace: dict | None = None,
    suffix: str = "",
) -> dict:
    source_run_id = f"sglang-{repeat_id}"
    run_id = f"{engine}-{repeat_id}{suffix}"
    if trace is None:
        trace = make_trace(repeat_id, source_run_id)
    if trace_path is None:
        trace_path = Path(f"/{repeat_id}.trace.json")
    trace_sha256 = trace["trace_sha256"]
    teacher_output_fingerprints = trace["output_fingerprints"]
    teacher_forcing_overhead = {
        "timed": True,
        "full_vocabulary_mask": False,
        "gpu_logit_elements_mutated_per_row": 0 if engine == "wkvm" else 1,
        "mutation": (
            "one_pending_token_scalar_overwrite"
            if engine == "wkvm"
            else "one_target_positive_infinity_scatter"
        ),
        "row_mutation_scope": (
            "request_loop" if engine == "wkvm" else "single_batched_scatter"
        ),
    }
    teacher_forcing_backend = {
        "wkvm": "post_sample_pending_token_override",
        "vllm": "vllm_sequence_logits_processor",
        "sglang": "sglang_sequence_logits_processor",
    }[engine]
    workload = {
        "sessions": SESSIONS,
        "turns": TURNS,
        "initial_context_tokens": CONTEXT,
        "turn_input_tokens": DELTA,
        "output_tokens_per_turn": OUTPUT,
        "required_model_len": REQUIRED_MODEL_LEN,
        "request_order_policy": "alternating",
        "request_order_seed": 0,
        "history_policy": "shared_teacher_forced_token_history",
        "history_trace_sha256": trace_sha256,
        "fingerprints": {
            **workload_fingerprints(),
            "teacher_forced_turn_outputs": teacher_output_fingerprints,
        },
    }
    request_ids = [f"session-{index:04d}" for index in range(SESSIONS)]
    continuation_turn_wall = (SESSIONS * OUTPUT) / rate
    turn_walls = [100.0] + [continuation_turn_wall] * (TURNS - 1)
    turns = []
    for index in range(TURNS):
        output_fingerprint = teacher_output_fingerprints[index]
        request_order = request_ids if index % 2 == 0 else list(reversed(request_ids))
        turns.append(
            {
                "turn_index": index,
                "wall_s": round(turn_walls[index], 6),
                "request_count": SESSIONS,
                "success_count": SESSIONS,
                "error_count": 0,
                "output_tokens": SESSIONS * OUTPUT,
                "output_fingerprint_complete": True,
                "generated_output_fingerprint": copy.deepcopy(output_fingerprint),
                "prompt_token_ids_sha256": stable_sha256(
                    f"prompt-{repeat_id}-{index}"
                ),
                "request_output_token_ids_sha256": output_fingerprint[
                    "request_output_token_ids_sha256"
                ],
                "request_order_policy": "alternating",
                "request_order": request_order,
                "requests": [
                    {
                        "session_id": request_id,
                        "success": True,
                        "error": None,
                        "output_tokens": OUTPUT,
                    }
                    for request_id in request_ids
                ],
                "teacher_forcing": {
                    "enabled": True,
                    "mode": "shared_teacher_forced",
                    "backend": teacher_forcing_backend,
                    "trace_sha256": trace_sha256,
                    "selected_outputs_match_trace": True,
                    "selected_output_exact_rows": SESSIONS,
                    "request_count": SESSIONS,
                    "teacher_output_fingerprint": copy.deepcopy(output_fingerprint),
                    "overhead_contract": teacher_forcing_overhead,
                },
                **(
                    {"reuse_invariants": {"passed": True}}
                    if engine == "wkvm"
                    else {}
                ),
            }
        )
    continuation_wall = sum(turn["wall_s"] for turn in turns[1:])
    continuation_output = SESSIONS * OUTPUT * (TURNS - 1)
    total_requests = SESSIONS * TURNS
    total_output = SESSIONS * OUTPUT * TURNS
    payload = {
        "schema": "wkvm.gemma_multiturn_bench.v1",
        "engine": engine,
        "engine_version": f"{engine}-test",
        "model_path": "/models/gemma-4-E4B-it",
        "dtype": "bfloat16",
        "prompt_token_source": "synthetic_lcg",
        "semantic_mode": (
            "routed_span_approximate" if engine == "wkvm" else "full_kv"
        ),
        "history_trace": {
            "mode": "shared_teacher_forced",
            "shared": True,
            "teacher_forced": True,
            "schema": "wkvm.gemma_shared_history_trace.v1",
            "trace_sha256": trace_sha256,
            "turn_count": TURNS,
            "output_fingerprints": teacher_output_fingerprints,
            "source_path": str(trace_path),
        },
        "benchmark_identity": {
            "campaign_id": CAMPAIGN_ID,
            "repeat_id": repeat_id,
            "run_id": run_id,
            "source_run_id": source_run_id,
            "memory_ceiling_mib": MEMORY_CEILING_MIB,
        },
        "git_tree_state": {"clean": True},
        "git_commit": GIT_COMMIT,
        "gpu_memory": {
            "baseline_used_mib": 512,
            "peak_used_mib": 20_000,
            "peak_delta_mib": 19_488,
            "memory_total_mib": 24_564,
            "device_uuid": "GPU-test",
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "query_error_count": 0,
            "error": None,
            "schema": "wkvm.whole_gpu_memory.v1",
            "scope": "whole_device",
            "source": "nvidia-smi",
            "sample_count": 100,
            "sample_interval_s": 0.1,
        },
        "workload": workload,
        "sampling": {
            "temperature": 0.0,
            "top_p": 1.0,
            "ignore_eos": True,
            "max_output_tokens_per_turn": OUTPUT,
            "teacher_forced": True,
        },
        "launch_config": {"environment": {}},
        "engine_config": {
            "teacher_forcing_backend": teacher_forcing_backend,
            "teacher_forcing_overhead_contract": teacher_forcing_overhead,
            **(
                {
                    "enable_token_pool_attention": True,
                    "token_pool_capacity": 131_072,
                    "token_pool_max_context_len": 37_632,
                    "slots": SESSIONS,
                }
                if engine == "wkvm"
                else {
                    "enable_prefix_caching": True,
                    "prefix_caching": True,
                    "max_model_len": 37_632,
                    "max_num_seqs": SESSIONS,
                    "capacity_telemetry": {
                        "kv_token_capacity": 140_640,
                        "kv_max_concurrency": 4.13,
                        "capacity_source": "vllm.cache_config",
                        "capacity_estimated": False,
                    },
                }
                if engine == "vllm"
                else {
                    "disable_radix_cache": False,
                    "context_length": 37_632,
                    "max_running_requests": SESSIONS,
                    "capacity_telemetry": {
                        "effective_token_capacity": 58_852,
                        "configured_max_running_requests": SESSIONS,
                        "capacity_source": "engine.get_server_info",
                        "capacity_error": None,
                    },
                }
            ),
        },
        "summary": {
            "requested_turns": TURNS,
            "completed_turn_rows": TURNS,
            "all_turns_recorded": True,
            "turn_rows": TURNS,
            "request_count": total_requests,
            "success_count": total_requests,
            "error_count": 0,
            "output_tokens": total_output,
            "continuation_turns": {
                "turn_rows": TURNS - 1,
                "request_count": SESSIONS * (TURNS - 1),
                "success_count": SESSIONS * (TURNS - 1),
                "error_count": 0,
                "output_tokens": continuation_output,
                "wall_s": round(continuation_wall, 6),
                "wall_scope": "sum_of_synchronized_engine_turn_barriers",
                "output_tok_s": round(continuation_output / continuation_wall, 3),
                "cache_telemetry_complete": True,
            },
        },
        "turns": turns,
        "engine_metrics_after_close": (
            {
                "fallback_decode_model_calls": 0,
                "mixed_batch_fallbacks": 0,
                "decode_batch_fallback_reasons": {},
                "token_pool_full_attention_coverage_splits": 0,
                "persistent_padded_decode_cuda_graph_skips": 0,
                "persistent_padded_decode_cuda_graph_skip_reasons": {},
                "token_pool_decode_graph_shape_mismatches": 0,
                "token_pool_decode_graph_shape_mismatch_reasons": {},
                "execution_mode": "mixed_ragged",
                "mixed_batch_model_calls": 1,
                "mixed_batch_opportunities": 1,
                "max_resident_sessions": SESSIONS,
                "max_resident_state_slots": SESSIONS,
                "token_pool": {
                    "enabled": True,
                    "token_slot_capacity": 131_072,
                },
            }
            if engine == "wkvm"
            else {}
        ),
    }
    return payload


def make_native_trace_source_artifact(
    engine: str,
    rate: float,
    **kwargs: object,
) -> dict:
    payload = make_artifact(engine, rate, **kwargs)
    emitted_trace = copy.deepcopy(payload["history_trace"])
    payload["history_trace"] = {
        "mode": "engine_generated",
        "shared": False,
        "teacher_forced": False,
    }
    payload["emitted_history_trace"] = emitted_trace
    payload["workload"]["history_policy"] = {
        "wkvm": "parked_state_plus_delta",
        "vllm": "cumulative_full_token_history",
        "sglang": "cumulative_full_token_history",
    }[engine]
    payload["workload"]["history_trace_sha256"] = None
    payload["workload"]["fingerprints"].pop(
        "teacher_forced_turn_outputs"
    )
    payload["sampling"] = {"teacher_forced": False}
    payload["sampling"].update(
        {
            "temperature": 0.0,
            "top_p": 1.0,
            "ignore_eos": True,
            "max_output_tokens_per_turn": OUTPUT,
        }
    )
    payload["engine_config"].update(
        {
            "history_mode": "engine_generated",
            "teacher_forcing_backend": None,
            "teacher_forcing_overhead_contract": None,
        }
    )
    for turn, output_fingerprint in zip(
        payload["turns"],
        emitted_trace["output_fingerprints"],
        strict=True,
    ):
        turn.update(
            {
                "request_count": SESSIONS,
                "success_count": SESSIONS,
                "error_count": 0,
                "output_fingerprint_complete": True,
                "generated_output_fingerprint": copy.deepcopy(
                    output_fingerprint
                ),
                "teacher_forcing": {
                    "enabled": False,
                    "mode": "engine_generated",
                },
            }
        )
    return payload


class TestMultiturn10xReport(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.output_directory = Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def strict_records(self) -> list[dict]:
        records = []
        for index in range(1, 4):
            repeat_id = f"r{index}"
            source_run_id = f"sglang-{repeat_id}"
            trace_path, trace = write_trace(
                self.output_directory,
                repeat_id,
                source_run_id,
            )
            records.extend(
                [
                    _artifact_record(
                        self.output_directory / f"wkvm-{repeat_id}.json",
                        make_artifact(
                            "wkvm",
                            1000.0 + index,
                            repeat_id=repeat_id,
                            trace_path=trace_path,
                            trace=trace,
                        ),
                    ),
                    _artifact_record(
                        self.output_directory / f"vllm-{repeat_id}.json",
                        make_artifact(
                            "vllm",
                            90.0 + index,
                            repeat_id=repeat_id,
                            trace_path=trace_path,
                            trace=trace,
                        ),
                    ),
                    _artifact_record(
                        self.output_directory / f"sglang-{repeat_id}.json",
                        make_native_trace_source_artifact(
                            "sglang",
                            80.0 + index,
                            repeat_id=repeat_id,
                            trace_path=trace_path,
                            trace=trace,
                        ),
                    ),
                ]
            )
        return records

    def test_conservative_gate_passes_only_against_both_incumbents(self) -> None:
        records = self.strict_records()

        report = build_report(records)

        self.assertTrue(report["passed"])
        self.assertGreaterEqual(report["ratios"]["vllm"], 10.0)
        self.assertGreaterEqual(report["ratios"]["sglang"], 10.0)
        self.assertEqual(
            report["target"]["continuation_output_tokens_per_turn"],
            1024,
        )
        self.assertEqual(report["ratio_witnesses"]["wkvm_min"]["repeat_id"], "r1")
        self.assertEqual(report["ratio_witnesses"]["vllm_max"]["repeat_id"], "r3")
        self.assertIn("Gate: PASS", render_markdown(report))

    def test_strict_gate_accepts_complete_publication_evidence(self) -> None:
        records = self.strict_records()
        self.assertEqual(
            len(
                {
                    record["raw_trace"]["logical_sha256"]
                    for record in records
                }
            ),
            3,
        )
        report = build_report(
            records,
            strict=True,
            whole_device_memory_ceiling_mib=24_000,
        )

        self.assertTrue(report["passed"])
        self.assertTrue(report["publication_passed"])
        self.assertTrue(all(report["publication_checks"].values()))
        self.assertIn("strict publication", render_markdown(report))

    def test_strict_gate_rejects_duplicate_paths_digests_and_run_ids(self) -> None:
        mutations = ("path", "payload_digest", "run_id")
        for field in mutations:
            with self.subTest(field=field):
                records = self.strict_records()
                records[1][field] = records[0][field]
                report = build_report(
                    records,
                    strict=True,
                    whole_device_memory_ceiling_mib=MEMORY_CEILING_MIB,
                )
                self.assertFalse(report["passed"])
                self.assertFalse(
                    report["publication_checks"]["unique_artifacts"]
                )

    def test_strict_gate_requires_exact_repeat_matrix(self) -> None:
        records = self.strict_records()
        records.pop()

        report = build_report(
            records,
            strict=True,
            whole_device_memory_ceiling_mib=MEMORY_CEILING_MIB,
        )

        self.assertFalse(report["passed"])
        self.assertFalse(report["publication_checks"]["exact_repeat_matrix"])

    def test_strict_gate_pairs_replays_to_native_source_run(self) -> None:
        records = self.strict_records()
        replay = next(
            record
            for record in records
            if record["repeat_id"] == "r2" and record["engine"] == "vllm"
        )
        replay["source_run_id"] = "wrong-source-run"

        report = build_report(
            records,
            strict=True,
            whole_device_memory_ceiling_mib=MEMORY_CEILING_MIB,
        )

        self.assertFalse(report["passed"])
        self.assertFalse(report["publication_checks"]["trace_role_contract"])

    def test_raw_trace_content_hash_is_recomputed(self) -> None:
        trace_path, trace = write_trace(
            self.output_directory,
            "r1",
            "sglang-r1",
        )
        trace["turn_outputs"][0][0][0] += 1
        trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True) + "\n")
        payload = make_artifact(
            "vllm",
            90.0,
            trace_path=trace_path,
            trace=trace,
        )

        record = _artifact_record(self.output_directory / "vllm.json", payload)

        self.assertFalse(record["raw_trace"]["valid"])

    def test_continuation_rate_and_tokens_are_recomputed(self) -> None:
        for mutation in ("rate", "tokens"):
            with self.subTest(mutation=mutation):
                payload = make_artifact("vllm", 90.0)
                if mutation == "rate":
                    payload["summary"]["continuation_turns"][
                        "output_tok_s"
                    ] = 99_999.0
                else:
                    payload["turns"][3]["output_tokens"] = 1
                record = _artifact_record("vllm.json", payload)
                self.assertIn(
                    "invalid_continuation_accounting",
                    record["errors"],
                )
                self.assertIsNone(record["continuation_rate"])

    def test_memory_delta_must_equal_peak_minus_baseline(self) -> None:
        payload = make_artifact("vllm", 90.0)
        payload["gpu_memory"]["peak_delta_mib"] = 0

        record = _artifact_record("vllm.json", payload)

        self.assertFalse(record["publication_checks"]["memory_delta"])

    def test_strict_gate_freezes_commit_model_version_and_config(self) -> None:
        fields = (
            ("git_commit", "e" * 40, "same_git_commit"),
            ("model_identity", ("/different/model", "bfloat16"), "stable_model_identity"),
            ("engine_version_identity", "different", "stable_engine_versions"),
            ("engine_config_signature", "different", "stable_engine_configs"),
        )
        for field, value, check in fields:
            with self.subTest(field=field):
                records = self.strict_records()
                records[0][field] = value
                report = build_report(
                    records,
                    strict=True,
                    whole_device_memory_ceiling_mib=MEMORY_CEILING_MIB,
                )
                self.assertFalse(report["passed"])
                self.assertFalse(report["publication_checks"][check])

    def test_strict_gate_requires_target_gpu_and_engine_limits(self) -> None:
        records = self.strict_records()
        records[0]["gpu_name"] = "NVIDIA A800"
        records[1]["publication_checks"]["engine_limits"] = False

        report = build_report(
            records,
            strict=True,
            whole_device_memory_ceiling_mib=MEMORY_CEILING_MIB,
        )

        self.assertFalse(report["passed"])
        self.assertFalse(report["publication_checks"]["expected_gpu"])
        self.assertFalse(report["publication_checks"]["engine_limits"])

    def test_engine_limit_check_uses_required_37600_tokens(self) -> None:
        payload = make_artifact("vllm", 90.0)
        payload["engine_config"]["max_model_len"] = REQUIRED_MODEL_LEN - 1

        record = _artifact_record("vllm.json", payload)

        self.assertFalse(record["publication_checks"]["engine_limits"])

    def test_identity_memory_ceiling_must_match_report_ceiling(self) -> None:
        records = self.strict_records()
        records[0]["identity_memory_ceiling_mib"] = MEMORY_CEILING_MIB - 1

        report = build_report(
            records,
            strict=True,
            whole_device_memory_ceiling_mib=MEMORY_CEILING_MIB,
        )

        self.assertFalse(report["passed"])
        self.assertFalse(report["publication_checks"]["identity_memory_ceiling"])

    def test_strict_gate_accepts_one_paired_native_source_repeat(self) -> None:
        trace_path, trace = write_trace(
            self.output_directory,
            "r1",
            "sglang-r1",
        )
        records = [
            _artifact_record(
                self.output_directory / "wkvm.json",
                make_artifact(
                    "wkvm",
                    1000.0,
                    trace_path=trace_path,
                    trace=trace,
                ),
            ),
            _artifact_record(
                self.output_directory / "vllm.json",
                make_artifact(
                    "vllm",
                    90.0,
                    trace_path=trace_path,
                    trace=trace,
                ),
            ),
            _artifact_record(
                self.output_directory / "sglang.json",
                make_native_trace_source_artifact(
                    "sglang",
                    80.0,
                    trace_path=trace_path,
                    trace=trace,
                ),
            ),
        ]

        report = build_report(
            records,
            min_repeats=1,
            strict=True,
            whole_device_memory_ceiling_mib=24_000,
        )

        self.assertTrue(report["passed"])
        self.assertTrue(report["checks"]["same_workload"])
        self.assertTrue(report["publication_checks"]["same_teacher_trace"])
        self.assertTrue(report["publication_checks"]["trace_role_contract"])
        self.assertEqual(
            report["engines"]["sglang"]["history_trace_roles"],
            ["native_trace_source"],
        )
        self.assertEqual(
            report["engines"]["vllm"]["history_trace_roles"],
            ["teacher_forced_replay"],
        )

    def test_native_trace_source_requires_exact_generated_fingerprints(self) -> None:
        payload = make_native_trace_source_artifact("wkvm", 1000.0)
        payload["turns"][3]["generated_output_fingerprint"] = {
            "request_output_token_ids_sha256": "different"
        }

        record = _artifact_record("wkvm.json", payload)

        self.assertEqual(record["history_trace_role"], "native_trace_source")
        self.assertFalse(record["publication_checks"]["fixed_history"])

    def test_native_trace_source_requires_zero_teacher_forcing_overhead(self) -> None:
        payload = make_native_trace_source_artifact("wkvm", 1000.0)
        payload["engine_config"]["teacher_forcing_backend"] = (
            "post_sample_pending_token_override"
        )

        record = _artifact_record("wkvm.json", payload)

        self.assertFalse(record["publication_checks"]["fixed_history"])
        self.assertFalse(
            record["publication_checks"]["bounded_teacher_forcing"]
        )

    def test_trace_role_contract_rejects_multiple_source_engines(self) -> None:
        records = [
            _artifact_record(
                "wkvm.json",
                make_native_trace_source_artifact("wkvm", 1000.0),
            ),
            _artifact_record(
                "vllm.json",
                make_native_trace_source_artifact("vllm", 90.0),
            ),
            _artifact_record("sglang.json", make_artifact("sglang", 80.0)),
        ]

        report = build_report(
            records,
            min_repeats=1,
            strict=True,
            whole_device_memory_ceiling_mib=24_000,
        )

        self.assertFalse(report["passed"])
        self.assertFalse(report["publication_checks"]["trace_role_contract"])

    def test_strict_gate_requires_explicit_common_memory_ceiling(self) -> None:
        missing = build_report(self.strict_records(), strict=True)
        exceeded = build_report(
            self.strict_records(),
            strict=True,
            whole_device_memory_ceiling_mib=19_000,
        )

        self.assertFalse(missing["passed"])
        self.assertFalse(
            missing["publication_checks"]["memory_ceiling_configured"]
        )
        self.assertFalse(missing["publication_checks"]["within_memory_ceiling"])
        self.assertFalse(exceeded["passed"])
        self.assertTrue(
            exceeded["publication_checks"]["memory_ceiling_configured"]
        )
        self.assertFalse(exceeded["publication_checks"]["within_memory_ceiling"])

    def test_memory_ceiling_cli_is_explicit(self) -> None:
        args = build_parser().parse_args(
            [
                "wkvm.json",
                "--strict",
                "--whole-device-memory-ceiling-mib",
                "24000",
            ]
        )

        self.assertTrue(args.strict)
        self.assertEqual(args.whole_device_memory_ceiling_mib, 24_000)

    def test_strict_cache_and_capacity_proofs_are_engine_specific(self) -> None:
        vllm_payload = make_artifact("vllm", 90.0)
        vllm_payload["engine_config"]["enable_prefix_caching"] = False
        vllm = _artifact_record("vllm.json", vllm_payload)
        sglang_payload = make_artifact("sglang", 80.0)
        sglang_payload["engine_config"]["capacity_telemetry"].pop(
            "effective_token_capacity"
        )
        sglang = _artifact_record("sglang.json", sglang_payload)

        self.assertFalse(vllm["publication_checks"]["cache_enabled"])
        self.assertTrue(vllm["publication_checks"]["capacity_telemetry"])
        self.assertTrue(sglang["publication_checks"]["cache_enabled"])
        self.assertFalse(sglang["publication_checks"]["capacity_telemetry"])

    def test_strict_gate_rejects_dirty_memory_and_fallback_evidence(self) -> None:
        records = self.strict_records()
        wkvm = next(record for record in records if record["engine"] == "wkvm")
        wkvm["publication_checks"].update(
            {
                "clean_worktree": False,
                "idle_gpu_baseline": False,
                "memory_delta": False,
                "no_fallbacks": False,
            }
        )

        report = build_report(
            records,
            strict=True,
            whole_device_memory_ceiling_mib=24_000,
        )

        self.assertFalse(report["passed"])
        self.assertFalse(report["publication_checks"]["clean_worktree"])
        self.assertFalse(report["publication_checks"]["idle_gpu_baseline"])
        self.assertFalse(report["publication_checks"]["memory_delta"])
        self.assertFalse(report["publication_checks"]["no_fallbacks"])

    def test_strict_gate_rejects_engine_local_prompt_histories(self) -> None:
        records = self.strict_records()
        vllm = next(record for record in records if record["engine"] == "vllm")
        vllm["history_trace_signature"] = ((0, "prompt-0"), (1, "different"))

        report = build_report(
            records,
            strict=True,
            whole_device_memory_ceiling_mib=24_000,
        )

        self.assertFalse(report["passed"])
        self.assertFalse(report["publication_checks"]["same_prompt_trace"])

    def test_strict_gate_rejects_engine_generated_history_metadata(self) -> None:
        payload = make_artifact("vllm", 90.0)
        payload["history_trace"] = {
            "mode": "engine_generated",
            "shared": False,
            "teacher_forced": False,
        }
        payload["workload"]["history_policy"] = (
            "cumulative_full_token_history"
        )

        record = _artifact_record("vllm.json", payload)

        self.assertFalse(record["publication_checks"]["fixed_history"])

    def test_strict_gate_requires_one_content_addressed_teacher_trace(self) -> None:
        records = self.strict_records()
        records[-1]["raw_trace"]["logical_sha256"] = "b" * 64

        report = build_report(
            records,
            strict=True,
            whole_device_memory_ceiling_mib=24_000,
        )

        self.assertFalse(report["passed"])
        self.assertFalse(report["publication_checks"]["same_teacher_trace"])

    def test_strict_gate_rejects_full_vocabulary_teacher_masking(self) -> None:
        payload = make_artifact("vllm", 90.0)
        payload["engine_config"]["teacher_forcing_overhead_contract"][
            "full_vocabulary_mask"
        ] = True

        record = _artifact_record("vllm.json", payload)

        self.assertFalse(
            record["publication_checks"]["bounded_teacher_forcing"]
        )

    def test_strict_gate_requires_wkvm_pending_state_override_proof(self) -> None:
        payload = make_artifact("wkvm", 1000.0)
        payload["engine_config"]["teacher_forcing_backend"] = (
            "vllm_sequence_logits_processor"
        )

        record = _artifact_record("wkvm.json", payload)

        self.assertFalse(record["publication_checks"]["fixed_history"])

    def test_execution_contract_is_required_only_by_strict_gate(self) -> None:
        records = self.strict_records()
        for record in records:
            if record["engine"] == "wkvm":
                record["publication_checks"]["execution_contract"] = False

        exploratory = build_report(records)
        strict = build_report(
            records,
            strict=True,
            whole_device_memory_ceiling_mib=24_000,
        )

        self.assertTrue(exploratory["passed"])
        self.assertFalse(exploratory["publication_passed"])
        self.assertFalse(strict["passed"])
        self.assertFalse(strict["publication_checks"]["execution_contract"])

    def test_zero_mixed_opportunities_require_partitioned_execution(self) -> None:
        payload = make_artifact("wkvm", 1000.0)
        metrics = payload["engine_metrics_after_close"]
        metrics["execution_mode"] = "partitioned_prefill_decode"
        metrics["mixed_batch_model_calls"] = 0
        metrics["mixed_batch_opportunities"] = 0

        record = _artifact_record("wkvm.json", payload)

        self.assertTrue(record["publication_checks"]["execution_contract"])

    def test_unexecuted_mixed_opportunity_fails_strict_check(self) -> None:
        payload = make_artifact("wkvm", 1000.0)
        metrics = payload["engine_metrics_after_close"]
        metrics["execution_mode"] = "partitioned_prefill_decode"
        metrics["mixed_batch_model_calls"] = 0
        metrics["mixed_batch_opportunities"] = 1

        record = _artifact_record("wkvm.json", payload)

        self.assertFalse(record["publication_checks"]["execution_contract"])

    def test_mixed_execution_requires_one_call_per_opportunity(self) -> None:
        payload = make_artifact("wkvm", 1000.0)
        metrics = payload["engine_metrics_after_close"]
        metrics["execution_mode"] = "mixed_ragged"
        metrics["mixed_batch_model_calls"] = 1
        metrics["mixed_batch_opportunities"] = 2

        record = _artifact_record("wkvm.json", payload)

        self.assertFalse(record["publication_checks"]["execution_contract"])

    def test_disabled_cuda_graphs_pass_when_skip_metrics_are_zero(self) -> None:
        payload = make_artifact("wkvm", 1000.0)
        payload["engine_config"]["persistent_padded_decode_cuda_graph"] = False

        record = _artifact_record("wkvm.json", payload)

        self.assertTrue(record["publication_checks"]["no_graph_skips"])

    def test_artifact_publication_checks_require_scoped_fixed_history(self) -> None:
        payload = make_artifact("wkvm", 1000.0)
        payload["semantic_mode"] = "full_kv"
        payload["workload"]["history_policy"] = "cumulative_full_token_history"
        payload["engine_metrics_after_close"][
            "persistent_padded_decode_cuda_graph_skips"
        ] = 1

        record = _artifact_record("wkvm.json", payload)

        self.assertFalse(record["publication_checks"]["semantic_scope"])
        self.assertFalse(record["publication_checks"]["fixed_history"])
        self.assertFalse(record["publication_checks"]["no_graph_skips"])

    def test_missing_runtime_telemetry_does_not_look_like_zero(self) -> None:
        payload = make_artifact("wkvm", 1000.0)
        payload.pop("engine_metrics_after_close")

        record = _artifact_record("wkvm.json", payload)

        self.assertFalse(record["publication_checks"]["no_fallbacks"])
        self.assertFalse(
            record["publication_checks"]["no_full_attention_coverage_splits"]
        )
        self.assertFalse(record["publication_checks"]["no_graph_skips"])

    def test_gate_fails_when_wkvm_misses_one_incumbent(self) -> None:
        records = []
        for index in range(3):
            records.extend(
                [
                    _artifact_record(
                        f"wkvm-{index}.json",
                        make_artifact("wkvm", 800.0),
                    ),
                    _artifact_record(
                        f"vllm-{index}.json",
                        make_artifact("vllm", 90.0),
                    ),
                    _artifact_record(
                        f"sglang-{index}.json",
                        make_artifact("sglang", 70.0),
                    ),
                ]
            )

        report = build_report(records)

        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["wkvm_vs_vllm"])
        self.assertTrue(report["checks"]["wkvm_vs_sglang"])

    def test_incomplete_or_mismatched_workloads_cannot_pass(self) -> None:
        records = [
            _artifact_record(
                "wkvm.json",
                make_artifact("wkvm", 1000.0),
            ),
            _artifact_record(
                "vllm.json",
                make_artifact("vllm", 1.0),
            ),
            _artifact_record(
                "sglang.json",
                make_artifact("sglang", 1.0),
            ),
        ]
        records[2]["workload_signature"] = ("different",)
        records[2]["structural_workload_signature"] = ("different",)
        records[0]["errors"] = ["request_errors"]

        report = build_report(records, min_repeats=1)

        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["same_workload"])
        self.assertFalse(report["checks"]["complete_artifacts"])


if __name__ == "__main__":
    unittest.main()
