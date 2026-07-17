import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
import uuid

from experiments.gemma_multiturn_bench import (
    _append_outputs,
    _turn_prompts_and_deltas,
    atomic_write_json,
    build_workload,
    summarize_run,
    summarize_turn,
    workload_fingerprints,
)
from experiments.multiturn_http_10x_report import (
    BENCH_SCHEMA,
    OUTPUT_FINGERPRINT_SCHEMA,
    SCHEMA,
    SOURCE_ROLE,
    TRACE_SCHEMA,
    artifact_record,
    build_report,
    load_records,
    main,
    render_markdown,
)


def _trace_sha(repeat_index: int, suffix: int = 0) -> str:
    return f"{repeat_index * 16 + suffix + 1:064x}"[-64:]


def _turn_outputs(
    repeat_index: int,
    turn_index: int,
    *,
    sessions: int,
    output_tokens: int,
    variant: int = 0,
):
    return [
        [
            100
            + repeat_index * 50
            + turn_index * 10
            + session_index * output_tokens
            + token_index
            + variant
            for token_index in range(output_tokens)
        ]
        for session_index in range(sessions)
    ]


def _gpu_runtime_telemetry(
    *,
    active_fraction: float = 0.95,
    sm_clock_mhz: float = 2520.0,
    gpu_utilization_percent: float = 96.0,
    power_draw_w: float = 410.0,
    temperature_gpu_c: float = 70.0,
):
    sample_count = 100
    active_sample_count = round(sample_count * active_fraction)

    def metric(value: float, count: int = sample_count):
        return {"count": count, "min": value, "mean": value, "max": value}

    return {
        "source": "same_nvidia_smi_samples_as_memory_monitor",
        "sample_count": sample_count,
        "active_sample_count": active_sample_count,
        "pstates": ["P2"],
        "active_pstates": ["P2"],
        "metrics": {
            "temperature_gpu_c": metric(temperature_gpu_c),
            "power_limit_w": metric(450.0),
        },
        "active_metrics": {
            "sm_clock_mhz": metric(sm_clock_mhz, active_sample_count),
            "gpu_utilization_percent": metric(
                gpu_utilization_percent,
                active_sample_count,
            ),
            "power_draw_w": metric(power_draw_w, active_sample_count),
        },
    }


def _artifact_payload(
    *,
    engine: str,
    repeat_index: int,
    continuation_rate: float,
    trace_sha256: str | None = None,
    source: bool = False,
    dirty: bool = False,
    peak_used_mib: float = 23_000,
    output_variant: int = 0,
    exact_outputs: bool = True,
    runtime_telemetry: dict | None = None,
    turn_0_wall_s: float = 1.0,
):
    sessions = 2
    turns = 3
    initial_context_tokens = 4
    turn_input_tokens = 2
    output_tokens = 2
    vocab_size = 512
    workload = build_workload(
        sessions=sessions,
        turns=turns,
        initial_context_tokens=initial_context_tokens,
        turn_input_tokens=turn_input_tokens,
        vocab_size=vocab_size,
    )
    histories = [list(prompt) for prompt in workload.initial_prompts]
    session_ids = [f"session-{index:04d}" for index in range(sessions)]
    rows = []
    generated_turns = []
    expected_turn_output = sessions * output_tokens
    continuation_wall = expected_turn_output / continuation_rate
    for turn_index in range(turns):
        prompts, deltas = _turn_prompts_and_deltas(
            workload,
            histories,
            turn_index,
        )
        outputs = _turn_outputs(
            repeat_index,
            turn_index,
            sessions=sessions,
            output_tokens=output_tokens,
            variant=output_variant,
        )
        row = summarize_turn(
            turn_index=turn_index,
            session_ids=session_ids,
            prompts=prompts,
            deltas=deltas,
            outputs=outputs,
            expected_output_tokens=output_tokens,
            new_input_tokens=(
                [len(prompt) for prompt in prompts]
                if turn_index == 0
                else [len(delta) for delta in deltas]
            ),
            wall_s=(
                turn_0_wall_s if turn_index == 0 else continuation_wall
            ),
            ttft_s=[0.05 + turn_index * 0.01 + index * 0.001 for index in range(sessions)],
            e2e_s=[0.20 + turn_index * 0.02 + index * 0.002 for index in range(sessions)],
            cached_tokens=[0 if turn_index == 0 else 10] * sessions,
            errors=[None] * sessions,
        )
        row["response_output_fingerprint"] = copy.deepcopy(
            row["generated_output_fingerprint"]
        )
        row["response_output_fingerprint_complete"] = exact_outputs
        row["response_token_ids_observed_count"] = (
            sessions if exact_outputs else 0
        )
        for request in row["requests"]:
            request.update(
                {
                    "observed_output_tokens": output_tokens,
                    "output_token_ids_observed": exact_outputs,
                    "output_token_ids_source": (
                        "response_token_ids"
                        if exact_outputs
                        else "count_only_without_token_ids"
                    ),
                }
            )
        rows.append(row)
        generated_turns.append(outputs)
        _append_outputs(histories, outputs)

    fingerprints = [row["generated_output_fingerprint"] for row in rows]
    trace_sha256 = trace_sha256 or _trace_sha(repeat_index)
    history_trace = {
        "mode": "engine_generated",
        "shared": False,
        "teacher_forced": False,
    }
    emitted_history_trace = None
    if source:
        emitted_history_trace = {
            "mode": "shared_teacher_forced",
            "shared": True,
            "teacher_forced": True,
            "schema": TRACE_SCHEMA,
            "trace_sha256": trace_sha256,
            "turn_count": turns,
            "output_fingerprints": fingerprints,
        }
    else:
        history_trace = {
            "mode": "shared_teacher_forced_http",
            "shared": True,
            "teacher_forced": True,
            "schema": TRACE_SCHEMA,
            "trace_sha256": trace_sha256,
            "turn_count": turns,
            "output_fingerprints": fingerprints,
        }
    workload_payload = {
        "sessions": sessions,
        "turns": turns,
        "initial_context_tokens": initial_context_tokens,
        "turn_input_tokens": turn_input_tokens,
        "output_tokens_per_turn": output_tokens,
        "required_model_len": (
            initial_context_tokens
            + turns * output_tokens
            + (turns - 1) * turn_input_tokens
        ),
        "history_policy": (
            "wkvm_token_session_initial_prompt_then_deltas"
            if engine == "wkvm"
            else "cumulative_full_token_history"
        ),
        "synchronized_turn_barriers": True,
        "request_order_policy": "alternating",
        "request_order_seed": 0,
        "fingerprints": workload_fingerprints(workload),
    }
    repeat_id = f"r{repeat_index}"
    run_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"http-report-{engine}-{repeat_id}-{trace_sha256}",
        )
    )
    payload = {
        "schema": BENCH_SCHEMA,
        "engine": engine,
        "semantic_mode": (
            "routed_span_approximate" if engine == "wkvm" else "full_kv"
        ),
        "model": "mock-gemma",
        "prompt_token_source": "synthetic_lcg",
        "benchmark_identity": {
            "campaign_id": "http-campaign",
            "repeat_id": repeat_id,
            "run_id": run_id,
            "artifact_role": SOURCE_ROLE if source else "http_teacher_forced_replay",
        },
        "history_trace": history_trace,
        "workload": workload_payload,
        "turns": rows,
        "summary": summarize_run(rows, turns),
        "gpu_memory": {
            "schema": "wkvm.whole_gpu_memory.v1",
            "scope": "whole_device",
            "peak_used_mib": peak_used_mib,
            "gpu_name": "Mock GPU",
            "device_uuid": "GPU-mock",
            "query_error_count": 0,
            "gpu_runtime_telemetry": (
                runtime_telemetry or _gpu_runtime_telemetry()
            ),
        },
        "git_tree_state": {
            "clean": not dirty,
            "tracked_clean": not dirty,
            "changed_path_count": 2 if dirty else 0,
            "tracked_changed_path_count": 1 if dirty else 0,
            "untracked_path_count": 1 if dirty else 0,
            "status_sha256": "a" * 64,
            "tracked_status_sha256": "b" * 64,
        },
    }
    if emitted_history_trace is not None:
        payload["emitted_history_trace"] = emitted_history_trace
    return payload


def _write_campaign(
    root: Path,
    *,
    repeats: int,
    rates=None,
    dirty: tuple[int, str] | None = None,
    memory=None,
    runtime_telemetry=None,
):
    rates = rates or {
        "wkvm": [110.0, 120.0, 115.0],
        "vllm": [9.0, 10.0, 9.5],
        "sglang": [5.0, 6.0, 5.5],
    }
    paths = []
    for repeat_index in range(1, repeats + 1):
        trace_sha256 = _trace_sha(repeat_index)
        for engine in ("wkvm", "vllm", "sglang"):
            payload = _artifact_payload(
                engine=engine,
                repeat_index=repeat_index,
                continuation_rate=rates[engine][repeat_index - 1],
                trace_sha256=trace_sha256,
                source=engine == "sglang",
                dirty=dirty == (repeat_index, engine),
                peak_used_mib=(
                    memory.get((repeat_index, engine), 23_000)
                    if memory
                    else 23_000
                ),
                runtime_telemetry=(
                    runtime_telemetry.get((repeat_index, engine))
                    if runtime_telemetry
                    else None
                ),
            )
            path = root / f"{engine}-{repeat_index}.json"
            atomic_write_json(path, payload)
            paths.append(path)
    return paths


class MultiTurnHttp10xReportTests(unittest.TestCase):
    def test_conservative_three_repeat_gate_passes_with_dirty_caveat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_campaign(
                Path(tmp),
                repeats=3,
                dirty=(2, "wkvm"),
            )
            report = build_report(
                load_records(paths),
                min_repeats=3,
                whole_device_memory_ceiling_mib=24_200,
            )

        self.assertEqual(report["schema"], SCHEMA)
        self.assertTrue(report["passed"])
        self.assertEqual(report["complete_repeat_count"], 3)
        self.assertAlmostEqual(report["conservative"]["wkvm_min_output_tok_s"], 110, places=2)
        self.assertAlmostEqual(report["conservative"]["vllm_max_output_tok_s"], 10, places=2)
        self.assertGreaterEqual(report["ratios"]["vllm"], 10)
        self.assertGreaterEqual(report["ratios"]["sglang"], 10)
        self.assertFalse(report["semantic_comparison"]["identical_modes"])
        self.assertTrue(
            any("semantic_mode_difference" in item for item in report["caveats"])
        )
        self.assertTrue(all(group["source_engine"] == "sglang" for group in report["repeat_groups"]))
        self.assertEqual(len(report["dirty_tree_artifacts"]), 1)
        markdown = render_markdown(report)
        self.assertIn("Dirty-tree caveats", markdown)
        self.assertIn("dirty", markdown)
        self.assertIn("**Gate: PASS**", markdown)
        self.assertIsNotNone(report["engines"]["wkvm"]["p95_e2e_s"]["max"])

    def test_slow_repeat_surfaces_non_gating_runtime_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rates = {
                "wkvm": [50.0, 100.0, 105.0],
                "vllm": [4.0, 4.0, 4.0],
                "sglang": [3.0, 3.0, 3.0],
            }
            telemetry = {
                (1, "wkvm"): _gpu_runtime_telemetry(
                    active_fraction=0.50,
                    sm_clock_mhz=2200.0,
                    gpu_utilization_percent=70.0,
                    power_draw_w=300.0,
                    temperature_gpu_c=80.0,
                )
            }
            paths = _write_campaign(
                Path(tmp),
                repeats=3,
                rates=rates,
                runtime_telemetry=telemetry,
            )
            report = build_report(
                load_records(paths),
                min_repeats=3,
                whole_device_memory_ceiling_mib=24_200,
            )

        diagnostics = report["stability"]["repeat_diagnostics"]
        slow = [item for item in diagnostics if item["slow_repeat_candidate"]]
        self.assertEqual(len(slow), 1)
        self.assertEqual(slow[0]["engine"], "wkvm")
        self.assertEqual(slow[0]["repeat_id"], "r1")
        self.assertIn("active_sm_clock_below_peer_median", slow[0]["signals"])
        self.assertIn(
            "active_sample_fraction_below_peer_median",
            slow[0]["signals"],
        )
        self.assertIn("gpu_temperature_above_peer_median", slow[0]["signals"])
        self.assertTrue(report["passed"])
        markdown = render_markdown(report)
        self.assertIn("Repeat stability diagnostics", markdown)
        self.assertIn("active_sm_clock_below_peer_median", markdown)

    def test_ratio_gate_uses_min_wkvm_over_max_incumbent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rates = {
                "wkvm": [90.0],
                "vllm": [10.0],
                "sglang": [5.0],
            }
            paths = _write_campaign(Path(tmp), repeats=1, rates=rates)
            report = build_report(load_records(paths), min_repeats=1)

        self.assertAlmostEqual(report["ratios"]["vllm"], 9.0, places=2)
        self.assertFalse(report["checks"]["wkvm_vs_vllm_10x"])
        self.assertFalse(report["passed"])

    def test_full_session_scope_gates_all_turn_wall_time(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for engine in ("wkvm", "vllm", "sglang"):
                payload = _artifact_payload(
                    engine=engine,
                    repeat_index=1,
                    continuation_rate={
                        "wkvm": 110.0,
                        "vllm": 10.0,
                        "sglang": 5.0,
                    }[engine],
                    source=engine == "sglang",
                    turn_0_wall_s={
                        "wkvm": 0.05,
                        "vllm": 10.0,
                        "sglang": 20.0,
                    }[engine],
                )
                path = root / f"{engine}.json"
                atomic_write_json(path, payload)
                paths.append(path)
            report = build_report(
                load_records(paths),
                min_repeats=1,
                claim_scope="full-session",
            )

        self.assertTrue(report["passed"])
        self.assertEqual(
            report["claim_scope"],
            "provider_http_complete_session_e2e",
        )
        self.assertGreater(report["ratios"]["vllm"], 10.0)
        self.assertGreater(report["ratios"]["sglang"], 10.0)
        self.assertAlmostEqual(
            report["engines"]["vllm"]["full_session_wall_s"]["min"],
            10.8,
        )
        markdown = render_markdown(report)
        self.assertIn("10x full-session gate", markdown)
        self.assertIn("provider_http_complete_session_e2e", markdown)
        self.assertIn("ratios apply only to the named modes", markdown)

    def test_output_fingerprint_fallback_links_source_and_replays(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for offset, engine in enumerate(("wkvm", "vllm", "sglang")):
                payload = _artifact_payload(
                    engine=engine,
                    repeat_index=1,
                    continuation_rate={"wkvm": 110, "vllm": 10, "sglang": 5}[engine],
                    trace_sha256=_trace_sha(1, offset),
                    source=engine == "sglang",
                )
                path = root / f"{engine}.json"
                atomic_write_json(path, payload)
                paths.append(path)
            report = build_report(load_records(paths), min_repeats=1)

        self.assertTrue(report["passed"])
        self.assertEqual(
            report["repeat_groups"][0]["linkage_method"],
            "per_turn_output_fingerprints",
        )
        self.assertTrue(any("trace_linkage_fallback" in item for item in report["caveats"]))

    def test_trace_and_output_mismatch_fails_linkage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for offset, engine in enumerate(("wkvm", "vllm", "sglang")):
                payload = _artifact_payload(
                    engine=engine,
                    repeat_index=1,
                    continuation_rate={"wkvm": 110, "vllm": 10, "sglang": 5}[engine],
                    trace_sha256=_trace_sha(1, offset),
                    source=engine == "sglang",
                    output_variant=1 if engine == "wkvm" else 0,
                )
                path = root / f"{engine}.json"
                atomic_write_json(path, payload)
                paths.append(path)
            report = build_report(load_records(paths), min_repeats=1)

        self.assertFalse(report["checks"]["trace_or_output_linkage"])
        self.assertFalse(report["passed"])

    def test_exact_output_and_memory_ceiling_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for engine in ("wkvm", "vllm", "sglang"):
                payload = _artifact_payload(
                    engine=engine,
                    repeat_index=1,
                    continuation_rate={"wkvm": 110, "vllm": 10, "sglang": 5}[engine],
                    source=engine == "sglang",
                    exact_outputs=engine != "vllm",
                    peak_used_mib=24_300 if engine == "wkvm" else 23_000,
                )
                path = root / f"{engine}.json"
                atomic_write_json(path, payload)
                paths.append(path)
            report = build_report(
                load_records(paths),
                min_repeats=1,
                whole_device_memory_ceiling_mib=24_200,
            )

        self.assertFalse(report["checks"]["exact_output_ids"])
        self.assertFalse(report["checks"]["within_memory_ceiling"])
        self.assertIn("inexact_output_ids", report["engines"]["vllm"]["errors"])
        self.assertFalse(report["passed"])

    def test_workload_identity_mismatch_fails_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for engine in ("wkvm", "vllm", "sglang"):
                payload = _artifact_payload(
                    engine=engine,
                    repeat_index=1,
                    continuation_rate={"wkvm": 110, "vllm": 10, "sglang": 5}[engine],
                    source=engine == "sglang",
                )
                if engine == "vllm":
                    payload["workload"]["turn_input_tokens"] = 3
                path = root / f"{engine}.json"
                atomic_write_json(path, payload)
                paths.append(path)
            report = build_report(load_records(paths), min_repeats=1)

        self.assertFalse(report["checks"]["workload_identity"])
        self.assertFalse(report["passed"])

    def test_min_repeats_and_cli_allow_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rates = {
                "wkvm": [90.0, 90.0],
                "vllm": [10.0, 10.0],
                "sglang": [5.0, 5.0],
            }
            paths = _write_campaign(root, repeats=2, rates=rates)
            report = build_report(load_records(paths), min_repeats=3)
            self.assertFalse(report["checks"]["minimum_repeats"])
            markdown_path = root / "report.md"
            summary_path = root / "summary.json"
            argv = [
                *(str(path) for path in paths),
                "--min-repeats",
                "2",
                "--markdown",
                str(markdown_path),
                "--summary-json",
                str(summary_path),
            ]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(argv), 1)
                self.assertEqual(main([*argv, "--allow-fail"]), 0)
            self.assertTrue(markdown_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertIn("**Gate: FAIL**", markdown_path.read_text())

    def test_artifact_schema_mismatch_is_reported(self) -> None:
        payload = _artifact_payload(
            engine="wkvm",
            repeat_index=1,
            continuation_rate=110,
        )
        payload["schema"] = "wrong"
        record = artifact_record("broken.json", payload)
        self.assertIn("schema_mismatch", record["errors"])


if __name__ == "__main__":
    unittest.main()
