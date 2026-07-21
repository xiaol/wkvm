import contextlib
import copy
import io
import json
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
from experiments.wkvm_serving_bench import (
    build_runtime_config_proof,
    build_target_server_launch_record,
    build_target_server_model_binding,
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
    engine_versions = {
        "wkvm": "git:" + "c" * 40,
        "vllm": "0.24.0",
        "sglang": "0.5.14",
    }
    engine_ports = {"wkvm": 8000, "vllm": 8001, "sglang": 8002}
    engine_version = engine_versions[engine]
    model_path = "/models/mock-gemma"
    model_identity = {
        "manifest_sha256": "e" * 64,
        "path": model_path,
        "served_name": "mock-gemma",
    }
    required_model_len = workload_payload["required_model_len"]
    if engine == "wkvm":
        target_server_config = {
            "batch_wait_s": 0.01,
            "continuation_prefill_microbatch_rows": sessions,
            "dtype": "bfloat16",
            "max_queue": sessions * 2,
            "model_identity": model_identity,
            "native_gemma_attention_backend": "triton_dense_gqa",
            "native_gemma_kv_sharing_fast_prefill": True,
            "native_gemma_projection_backend": "separate",
            "persistent_padded_decode_steps": output_tokens,
            "prefill_microbatch_rows": 2,
            "slots": sessions,
            "token_pool_capacity": 4096,
            "token_pool_max_context_len": required_model_len,
            "token_pool_paged_block_size": 16,
        }
        launch_command = (
            "setsid env CUDA_VISIBLE_DEVICES=0 python -m wkvm.gemma_server "
            f"--model {model_path} --served-model-name mock-gemma "
            f"--host 127.0.0.1 --port {engine_ports[engine]} "
            f"--batch-wait-s 0.01 --continuation-prefill-microbatch-rows {sessions} "
            f"--max-queue {sessions * 2} --native-gemma-attention-backend "
            "triton_dense_gqa --native-gemma-kv-sharing-fast-prefill "
            "--native-gemma-projection-backend separate "
            f"--persistent-padded-decode-steps {output_tokens} "
            f"--prefill-microbatch-rows 2 --slots {sessions} "
            f"--token-pool-capacity 4096 --token-pool-max-context-len {required_model_len} "
            "--token-pool-paged-block-size 16"
        )
        server_metrics = {
            "engine": {
                "continuation_prefill_microbatch_rows": sessions,
                "max_resident_state_slots": sessions,
                "native_gemma_attention_backend": "triton_dense_gqa",
                "native_gemma_kv_sharing_fast_prefill": True,
                "native_gemma_projection_backend": "separate",
                "persistent_padded_decode": True,
                "persistent_padded_decode_steps": output_tokens,
                "prefill_microbatch_rows": 2,
                "token_pool_attention_enabled": True,
                "token_pool": {
                    "enabled": True,
                    "attention_enabled": True,
                    "token_slot_capacity": 4096,
                    "max_context_len": required_model_len,
                    "paged_block_size": 16,
                },
            },
            "server": {
                "batch_wait_s": 0.01,
                "max_chat_sessions": sessions,
                "max_queue": sessions * 2,
            },
        }
    elif engine == "vllm":
        compilation = {
            "mode": 3,
            "cudagraph_mode": "FULL_AND_PIECEWISE",
            "cudagraph_capture_sizes": [1, 2],
        }
        target_server_config = {
            "compilation_config": compilation,
            "dtype": "bfloat16",
            "enable_chunked_prefill": True,
            "enable_prefix_caching": True,
            "gpu_memory_utilization": 0.92,
            "kv_sharing_fast_prefill": True,
            "max_model_len": required_model_len,
            "max_num_batched_tokens": 1024,
            "max_num_seqs": sessions,
            "model_identity": model_identity,
            "model_runner_generation": "v1",
        }
        compilation_json = json.dumps(compilation, separators=(",", ":"))
        launch_command = (
            "setsid env CUDA_VISIBLE_DEVICES=0 VLLM_USE_V2_MODEL_RUNNER=0 "
            f"python -m vllm.entrypoints.openai.api_server --model {model_path} "
            f"--served-model-name mock-gemma --host 127.0.0.1 --port {engine_ports[engine]} "
            f"--dtype bfloat16 --max-model-len {required_model_len} "
            f"--max-num-seqs {sessions} --gpu-memory-utilization 0.92 "
            "--max-num-batched-tokens 1024 --enable-chunked-prefill "
            "--enable-prefix-caching --kv-sharing-fast-prefill "
            f"--compilation-config '{compilation_json}'"
        )
        server_metrics = {
            "vllm_config": {
                "model_config": {"model": model_path, "dtype": "bfloat16", "max_model_len": required_model_len},
                "cache_config": {
                    "gpu_memory_utilization": 0.92,
                    "enable_prefix_caching": True,
                    "kv_sharing_fast_prefill": True,
                    "kv_cache_size_tokens": 4096,
                    "kv_cache_max_concurrency": 2.0,
                },
                "scheduler_config": {
                    "max_num_batched_tokens": 1024,
                    "max_num_seqs": sessions,
                    "enable_chunked_prefill": True,
                },
                "compilation_config": compilation,
            },
            "vllm_env": {"VLLM_USE_V2_MODEL_RUNNER": False},
        }
    else:
        target_server_config = {
            "attention_backend": "triton",
            "chunked_prefill_size": 1024,
            "context_length": required_model_len,
            "cuda_graph_backend_decode": "full",
            "cuda_graph_backend_prefill": "disabled",
            "dtype": "bfloat16",
            "max_running_requests": sessions,
            "max_total_tokens": 4096,
            "mem_fraction_static": 0.92,
            "model_identity": model_identity,
        }
        launch_command = (
            "setsid env CUDA_VISIBLE_DEVICES=0 sglang serve "
            f"--model {model_path} --served-model-name mock-gemma "
            f"--host 127.0.0.1 --port {engine_ports[engine]} --dtype bfloat16 "
            f"--attention-backend triton --chunked-prefill-size 1024 "
            f"--context-length {required_model_len} --cuda-graph-backend-decode full "
            "--cuda-graph-backend-prefill disabled "
            f"--max-running-requests {sessions} --max-total-tokens 4096 "
            "--mem-fraction-static 0.92"
        )
        server_metrics = {
            "attention_backend": "triton",
            "chunked_prefill_size": 1024,
            "context_length": required_model_len,
            "cuda_graph_backend_decode": "full",
            "cuda_graph_backend_prefill": "disabled",
            "disable_cuda_graph": False,
            "disable_decode_cuda_graph": False,
            "disable_overlap_schedule": False,
            "disable_radix_cache": False,
            "disable_chunked_prefix_cache": False,
            "enable_torch_compile": False,
            "enable_two_batch_overlap": False,
            "dtype": "bfloat16",
            "max_running_requests": sessions,
            "max_total_tokens": 4096,
            "max_total_num_tokens": 4096,
            "mem_fraction_static": 0.92,
            "model_path": model_path,
            "internal_states": [
                {"effective_max_running_requests_per_dp": sessions}
            ],
        }
    base_url = f"http://127.0.0.1:{engine_ports[engine]}"
    launch_record = build_target_server_launch_record(
        launch_command,
        base_url=base_url,
        gpu_selector="0",
    )
    model_binding = build_target_server_model_binding(
        launch_command,
        target_server_config,
        served_model="mock-gemma",
    )
    runtime_config_proof = build_runtime_config_proof(
        engine,
        target_server_config,
        server_metrics,
        workload=workload_payload,
    )
    payload = {
        "schema": BENCH_SCHEMA,
        "engine": engine,
        "engine_version": engine_version,
        "semantic_mode": (
            "routed_span_approximate" if engine == "wkvm" else "full_kv"
        ),
        "model": "mock-gemma",
        "api": {
            "base_url": base_url,
            "endpoint": "/v1/completions",
        },
        "git_commit": "c" * 40,
        "provenance": {
            "engine": {
                "label": engine,
                "version": engine_version,
                "version_source": "frozen_campaign",
            },
            "target_server": {
                "launch_command": launch_command,
                "launch_command_source": "operator_supplied",
                "launch_profile": "legacy-untrusted-profile",
                "launch_profile_source": "operator_supplied_untrusted",
                "launch_argv": launch_record,
                "launch_argv_source": "derived_from_launch_command",
                "config": target_server_config,
                "config_source": "operator_supplied",
                "model_binding": model_binding,
            },
        },
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
        "server_metrics_after_run": server_metrics,
        "server_metrics_error": None,
        "runtime_config_proof": runtime_config_proof,
        "gpu_memory": {
            "schema": "wkvm.whole_gpu_memory.v1",
            "scope": "whole_device",
            "source": "nvidia-smi",
            "device_selector": "0",
            "baseline_used_mib": 512,
            "peak_used_mib": peak_used_mib,
            "peak_delta_mib": peak_used_mib - 512,
            "memory_total_mib": 24_564,
            "gpu_name": "Mock GPU",
            "device_uuid": "GPU-mock",
            "driver_version": "595.58.03",
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


def _refresh_publication_proof(payload: dict) -> None:
    target = payload["provenance"]["target_server"]
    target["launch_argv"] = build_target_server_launch_record(
        target["launch_command"],
        base_url=payload["api"]["base_url"],
        gpu_selector=payload["gpu_memory"]["device_selector"],
    )
    target["model_binding"] = build_target_server_model_binding(
        target["launch_command"],
        target["config"],
        served_model=payload["model"],
    )
    payload["runtime_config_proof"] = build_runtime_config_proof(
        payload["engine"],
        target["config"],
        payload["server_metrics_after_run"],
        workload=payload["workload"],
    )


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
    def test_strict_publication_gate_accepts_clean_paired_three_repeat_campaign(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_campaign(Path(tmp), repeats=3)
            report = build_report(
                load_records(paths),
                min_repeats=3,
                whole_device_memory_ceiling_mib=24_200,
                strict=True,
            )

        self.assertTrue(report["passed"])
        self.assertTrue(report["publication_passed"])
        self.assertTrue(all(report["publication_checks"].values()))
        self.assertEqual(report["model_manifest_sha256"], "e" * 64)
        self.assertTrue(
            report["publication_checks"]["same_model_manifest_sha256"]
        )
        self.assertEqual(
            {item["model_manifest_sha256"] for item in report["artifacts"]},
            {"e" * 64},
        )
        self.assertIn(
            f"Model manifest SHA-256: `{'e' * 64}`.",
            render_markdown(report),
        )

    def test_strict_publication_rejects_missing_or_cross_engine_model_manifest(
        self,
    ) -> None:
        for mutation in ("missing", "different"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                paths = _write_campaign(Path(tmp), repeats=3)
                for path in paths:
                    payload = json.loads(path.read_text())
                    if payload["engine"] != "vllm":
                        continue
                    model_identity = payload["provenance"]["target_server"][
                        "config"
                    ]["model_identity"]
                    if mutation == "missing":
                        model_identity.pop("manifest_sha256")
                    else:
                        model_identity["manifest_sha256"] = "d" * 64
                    atomic_write_json(path, payload)
                report = build_report(
                    load_records(paths),
                    min_repeats=3,
                    whole_device_memory_ceiling_mib=24_200,
                    strict=True,
                )

            self.assertFalse(report["passed"])
            self.assertFalse(report["publication_passed"])
            self.assertFalse(
                report["publication_checks"]["same_model_manifest_sha256"]
            )
            self.assertTrue(report["publication_checks"]["stable_engine_configs"])
            self.assertIsNone(report["model_manifest_sha256"])
            self.assertIn(
                "Model manifest SHA-256: unverified (missing or inconsistent).",
                render_markdown(report),
            )

    def test_strict_publication_requires_three_repeats_and_exact_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_campaign(Path(tmp), repeats=1)
            report = build_report(
                load_records(paths),
                min_repeats=1,
                whole_device_memory_ceiling_mib=24_200,
                strict=True,
            )

        self.assertFalse(report["passed"])
        self.assertFalse(report["publication_checks"]["minimum_repeats"])
        self.assertFalse(report["publication_checks"]["exact_repeat_matrix"])

    def test_strict_publication_rejects_dirty_tree_and_mixed_gpu_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_campaign(
                Path(tmp),
                repeats=3,
                dirty=(2, "wkvm"),
            )
            payload = json.loads(paths[4].read_text())
            payload["gpu_memory"]["device_uuid"] = "GPU-other"
            paths[4].write_text(json.dumps(payload))
            report = build_report(
                load_records(paths),
                min_repeats=3,
                whole_device_memory_ceiling_mib=24_200,
                strict=True,
            )

        self.assertFalse(report["passed"])
        self.assertFalse(report["publication_checks"]["clean_worktree"])
        self.assertFalse(report["publication_checks"]["same_gpu"])

    def test_homogeneous_pool_accepts_one_paired_gpu_per_repeat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_campaign(Path(tmp), repeats=3)
            for repeat_index in range(1, 4):
                for engine in ("wkvm", "vllm", "sglang"):
                    path = Path(tmp) / f"{engine}-{repeat_index}.json"
                    payload = json.loads(path.read_text())
                    payload["gpu_memory"]["device_uuid"] = f"GPU-pool-{repeat_index}"
                    atomic_write_json(path, payload)
            report = build_report(
                load_records(paths),
                min_repeats=3,
                whole_device_memory_ceiling_mib=24_200,
                gpu_policy="homogeneous-pool",
                strict=True,
            )

        self.assertTrue(report["passed"])
        self.assertTrue(report["publication_checks"]["same_gpu"])
        self.assertTrue(report["publication_checks"]["paired_repeat_gpu"])
        self.assertTrue(report["publication_checks"]["homogeneous_gpu_pool"])
        self.assertEqual(
            report["gpu_policy_details"]["repeat_gpu_uuids"]["r2"],
            ["GPU-pool-2"],
        )

    def test_homogeneous_pool_rejects_cross_engine_gpu_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_campaign(Path(tmp), repeats=3)
            for path in paths:
                payload = json.loads(path.read_text())
                repeat_index = int(payload["benchmark_identity"]["repeat_id"][1:])
                payload["gpu_memory"]["device_uuid"] = f"GPU-pool-{repeat_index}"
                atomic_write_json(path, payload)
            mismatch = Path(tmp) / "vllm-2.json"
            payload = json.loads(mismatch.read_text())
            payload["gpu_memory"]["device_uuid"] = "GPU-wrong"
            atomic_write_json(mismatch, payload)
            report = build_report(
                load_records(paths),
                min_repeats=3,
                whole_device_memory_ceiling_mib=24_200,
                gpu_policy="homogeneous-pool",
                strict=True,
            )

        self.assertFalse(report["passed"])
        self.assertFalse(report["publication_checks"]["same_gpu"])
        self.assertFalse(report["publication_checks"]["paired_repeat_gpu"])

    def test_homogeneous_pool_rejects_hardware_or_driver_drift(self) -> None:
        for field, value in (
            ("gpu_name", "Different GPU"),
            ("memory_total_mib", 48_000),
            ("driver_version", "999.0"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                paths = _write_campaign(Path(tmp), repeats=3)
                for path in paths:
                    payload = json.loads(path.read_text())
                    repeat_index = int(
                        payload["benchmark_identity"]["repeat_id"][1:]
                    )
                    payload["gpu_memory"]["device_uuid"] = (
                        f"GPU-pool-{repeat_index}"
                    )
                    atomic_write_json(path, payload)
                drift = Path(tmp) / "wkvm-3.json"
                payload = json.loads(drift.read_text())
                payload["gpu_memory"][field] = value
                atomic_write_json(drift, payload)
                report = build_report(
                    load_records(paths),
                    min_repeats=3,
                    whole_device_memory_ceiling_mib=24_200,
                    gpu_policy="homogeneous-pool",
                    strict=True,
                )

            self.assertFalse(report["publication_checks"]["same_gpu"])
            self.assertFalse(
                report["publication_checks"]["homogeneous_gpu_pool"]
            )

    def test_pool_launch_identity_derives_from_argv_and_ignores_claimed_profile(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_campaign(Path(tmp), repeats=3)
            for path in paths:
                payload = json.loads(path.read_text())
                repeat_index = int(payload["benchmark_identity"]["repeat_id"][1:])
                payload["gpu_memory"]["device_uuid"] = f"GPU-pool-{repeat_index}"
                payload["gpu_memory"]["device_selector"] = str(repeat_index)
                target = payload["provenance"]["target_server"]
                target["launch_command"] = target["launch_command"].replace(
                    "CUDA_VISIBLE_DEVICES=0",
                    f"CUDA_VISIBLE_DEVICES={repeat_index}",
                )
                old_port = payload["api"]["base_url"].rsplit(":", 1)[1]
                new_port = str(9000 + repeat_index)
                target["launch_command"] = target["launch_command"].replace(
                    f"--port {old_port}", f"--port {new_port}"
                )
                payload["api"]["base_url"] = f"http://127.0.0.1:{new_port}"
                _refresh_publication_proof(payload)
                atomic_write_json(path, payload)
            passing = build_report(
                load_records(paths),
                min_repeats=3,
                whole_device_memory_ceiling_mib=24_200,
                gpu_policy="homogeneous-pool",
                strict=True,
            )
            drift = Path(tmp) / "vllm-3.json"
            payload = json.loads(drift.read_text())
            payload["provenance"]["target_server"]["launch_profile"] += (
                " --different-engine-setting"
            )
            atomic_write_json(drift, payload)
            untrusted_profile = build_report(
                load_records(paths),
                min_repeats=3,
                whole_device_memory_ceiling_mib=24_200,
                gpu_policy="homogeneous-pool",
                strict=True,
            )
            payload["provenance"]["target_server"]["launch_command"] = payload[
                "provenance"
            ]["target_server"]["launch_command"].replace(
                "--dtype bfloat16", "--dtype float16"
            )
            atomic_write_json(drift, payload)
            failing = build_report(
                load_records(paths),
                min_repeats=3,
                whole_device_memory_ceiling_mib=24_200,
                gpu_policy="homogeneous-pool",
                strict=True,
            )

        self.assertTrue(passing["publication_checks"]["stable_engine_launches"])
        self.assertTrue(
            untrusted_profile["publication_checks"]["stable_engine_launches"]
        )
        self.assertTrue(untrusted_profile["passed"])
        self.assertFalse(failing["publication_checks"]["stable_engine_launches"])
        self.assertFalse(failing["publication_checks"]["launch_argv_binding"])

    def test_strict_publication_rejects_commit_model_and_driver_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_campaign(Path(tmp), repeats=3)
            mutations = (
                (paths[0], "git_commit", "d" * 40),
                (paths[1], "model", "other-gemma"),
            )
            for path, field, value in mutations:
                payload = json.loads(path.read_text())
                payload[field] = value
                path.write_text(json.dumps(payload))
            payload = json.loads(paths[2].read_text())
            payload["gpu_memory"]["driver_version"] = "999.0"
            paths[2].write_text(json.dumps(payload))
            report = build_report(
                load_records(paths),
                min_repeats=3,
                whole_device_memory_ceiling_mib=24_200,
                strict=True,
            )

        self.assertFalse(report["passed"])
        self.assertFalse(report["publication_checks"]["same_git_commit"])
        self.assertFalse(report["publication_checks"]["stable_model_identity"])
        self.assertFalse(report["publication_checks"]["same_driver"])

    def test_strict_publication_rejects_engine_version_or_source_drift(self) -> None:
        for field, value in (
            ("version", "0.24.1"),
            ("version_source", "different_environment"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                paths = _write_campaign(Path(tmp), repeats=3)
                payload = json.loads(paths[4].read_text())
                payload["provenance"]["engine"][field] = value
                if field == "version":
                    payload["engine_version"] = value
                paths[4].write_text(json.dumps(payload))
                report = build_report(
                    load_records(paths),
                    min_repeats=3,
                    whole_device_memory_ceiling_mib=24_200,
                    strict=True,
                )

                self.assertFalse(report["passed"])
                self.assertFalse(
                    report["publication_checks"]["stable_engine_versions"]
                )

    def test_strict_publication_rejects_target_server_launch_or_config_drift(
        self,
    ) -> None:
        for field in ("launch_command", "config"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                paths = _write_campaign(Path(tmp), repeats=3)
                payload = json.loads(paths[4].read_text())
                target = payload["provenance"]["target_server"]
                if field == "launch_command":
                    target[field] += " --changed"
                    failed_check = "stable_engine_launches"
                else:
                    target[field]["changed"] = True
                    failed_check = "stable_engine_configs"
                paths[4].write_text(json.dumps(payload))
                report = build_report(
                    load_records(paths),
                    min_repeats=3,
                    whole_device_memory_ceiling_mib=24_200,
                    strict=True,
                )

                self.assertFalse(report["passed"])
                self.assertFalse(report["publication_checks"][failed_check])

    def test_strict_publication_requires_target_server_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_campaign(Path(tmp), repeats=3)
            payload = json.loads(paths[4].read_text())
            payload["provenance"].pop("target_server")
            paths[4].write_text(json.dumps(payload))
            report = build_report(
                load_records(paths),
                min_repeats=3,
                whole_device_memory_ceiling_mib=24_200,
                strict=True,
            )

        self.assertFalse(report["passed"])
        self.assertFalse(
            report["publication_checks"]["target_server_provenance"]
        )

    def test_trace_sha_alone_cannot_bind_different_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = []
            for engine in ("wkvm", "vllm", "sglang"):
                payload = _artifact_payload(
                    engine=engine,
                    repeat_index=1,
                    continuation_rate={
                        "wkvm": 110,
                        "vllm": 10,
                        "sglang": 5,
                    }[engine],
                    trace_sha256=_trace_sha(1),
                    source=engine == "sglang",
                    output_variant=1 if engine == "wkvm" else 0,
                )
                path = root / f"{engine}.json"
                atomic_write_json(path, payload)
                paths.append(path)
            records = load_records(paths)
            report = build_report(records, min_repeats=1)

        self.assertFalse(report["checks"]["trace_or_output_linkage"])
        self.assertFalse(report["repeat_groups"][0]["complete"])

    def test_strict_publication_binds_trace_metadata_to_measured_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _write_campaign(Path(tmp), repeats=3)
            payload = json.loads(paths[1].read_text())
            payload["history_trace"]["output_fingerprints"][0][
                "request_output_token_ids_sha256"
            ] = "f" * 64
            paths[1].write_text(json.dumps(payload))
            report = build_report(
                load_records(paths),
                min_repeats=3,
                whole_device_memory_ceiling_mib=24_200,
                strict=True,
            )

        self.assertFalse(report["passed"])
        self.assertFalse(report["publication_checks"]["trace_output_binding"])
        self.assertFalse(report["publication_checks"]["trace_role_contract"])

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
