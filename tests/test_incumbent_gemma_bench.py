from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments.incumbent_gemma_bench import (
    ResidencyTelemetryMonitor,
    TelemetryUnavailable,
    VramMonitor,
    make_row,
    measure_vllm_generation,
    residency_telemetry_row_fields,
    row_green,
    sglang_capacity_telemetry,
    sglang_language_model_override,
    sglang_output_retractions,
    sglang_runtime_telemetry_sample,
    vllm_capacity_telemetry,
    vllm_request_metrics_timing,
    vllm_runtime_telemetry_sample,
)
from experiments.sglang_gemma_server import (
    build_arg_parser as build_sglang_server_arg_parser,
    server_kwargs as sglang_server_kwargs,
)


class TestIncumbentGemmaBench(unittest.TestCase):
    @staticmethod
    def _vllm_output(
        token_ids: list[int],
        *,
        first_token_ts: float | None,
        last_token_ts: float | None,
        first_token_latency: float | None = None,
    ) -> SimpleNamespace:
        metrics = None
        if first_token_ts is not None or last_token_ts is not None:
            metrics = SimpleNamespace(
                first_token_ts=first_token_ts,
                last_token_ts=last_token_ts,
                first_token_latency=first_token_latency,
            )
        return SimpleNamespace(
            outputs=[SimpleNamespace(token_ids=token_ids)],
            metrics=metrics,
        )

    def test_vllm_request_metrics_timing_uses_batch_wide_same_run_interval(
        self,
    ) -> None:
        outputs = [
            self._vllm_output(
                [1, 2, 3],
                first_token_ts=12.0,
                last_token_ts=16.0,
                first_token_latency=2.0,
            ),
            self._vllm_output(
                [4, 5, 6],
                first_token_ts=10.0,
                last_token_ts=15.0,
                first_token_latency=1.5,
            ),
        ]

        timing = vllm_request_metrics_timing(outputs, expected=2)

        self.assertIsNotNone(timing)
        assert timing is not None
        self.assertEqual(timing["decode_seconds"], 6.0)
        self.assertEqual(timing["prefill_plus_first_s"], 1.5)
        self.assertEqual(timing["decode_timing_method"], "same_run_request_metrics")
        self.assertTrue(timing["decode_timing_comparable"])
        self.assertEqual(timing["decode_timing_request_count"], 2)

    def test_measure_vllm_generation_skips_one_token_run_with_exact_metrics(
        self,
    ) -> None:
        outputs = [
            self._vllm_output(
                [1, 2, 3],
                first_token_ts=10.0,
                last_token_ts=12.5,
                first_token_latency=1.0,
            )
        ]
        llm = mock.Mock()
        llm.generate.return_value = outputs
        sp1 = object()
        spn = object()

        with (
            mock.patch(
                "experiments.incumbent_gemma_bench.synchronize_cuda"
            ),
            mock.patch(
                "experiments.incumbent_gemma_bench.time.perf_counter",
                side_effect=[100.0, 104.0],
            ),
        ):
            measured_outputs, wall_s, timing = measure_vllm_generation(
                llm,
                [{"prompt_token_ids": [1]}],
                sp1,
                spn,
            )

        self.assertIs(measured_outputs, outputs)
        self.assertEqual(wall_s, 4.0)
        self.assertEqual(timing["decode_seconds"], 2.5)
        self.assertTrue(timing["decode_timing_comparable"])
        llm.generate.assert_called_once_with(
            [{"prompt_token_ids": [1]}], spn, use_tqdm=False
        )

    def test_measure_vllm_generation_labels_separate_run_fallback(self) -> None:
        full_outputs = [
            self._vllm_output(
                [1, 2, 3],
                first_token_ts=None,
                last_token_ts=None,
            )
        ]
        llm = mock.Mock()
        llm.generate.side_effect = [full_outputs, []]
        sp1 = object()
        spn = object()

        with (
            mock.patch(
                "experiments.incumbent_gemma_bench.synchronize_cuda"
            ),
            mock.patch(
                "experiments.incumbent_gemma_bench.time.perf_counter",
                side_effect=[100.0, 110.0, 200.0, 203.0],
            ),
        ):
            _, wall_s, timing = measure_vllm_generation(
                llm,
                [{"prompt_token_ids": [1]}],
                sp1,
                spn,
            )

        self.assertEqual(wall_s, 10.0)
        self.assertEqual(timing["decode_seconds"], 7.0)
        self.assertEqual(timing["decode_timing_method"], "separate_run_subtraction")
        self.assertFalse(timing["decode_timing_comparable"])
        self.assertIn("not directly comparable", timing["decode_timing_note"])
        self.assertEqual(
            llm.generate.call_args_list,
            [
                mock.call([{"prompt_token_ids": [1]}], spn, use_tqdm=False),
                mock.call([{"prompt_token_ids": [1]}], sp1, use_tqdm=False),
            ],
        )

    def test_vram_monitor_emits_shared_memory_schema(self) -> None:
        monitor = VramMonitor(interval_s=0.25)
        monitor.baseline_mib = 1024
        monitor.peak_mib = 4096

        result = monitor.result()

        self.assertEqual(result["schema"], "wkvm.whole_gpu_memory.v1")
        self.assertEqual(result["scope"], "whole_device")
        self.assertEqual(result["baseline_used_mib"], 1024)
        self.assertEqual(result["peak_used_mib"], 4096)
        self.assertEqual(result["peak_delta_mib"], 3072)

    def test_vram_monitor_parses_external_sampler_output(self) -> None:
        monitor = VramMonitor(interval_s=0.1)
        monitor.baseline_mib = 1024
        monitor.peak_mib = 1024

        monitor._record_samples("2048\n8192\ninvalid\n4096\n")

        result = monitor.result()
        self.assertEqual(result["peak_used_mib"], 8192)
        self.assertEqual(result["peak_delta_mib"], 7168)
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["query_error_count"], 1)
        self.assertIn("unexpected nvidia-smi sample", result["error"])

    def test_make_row_fingerprints_generated_token_ids(self) -> None:
        row = make_row(
            B=2,
            prompt_lens=[4, 4],
            first_wall_s=1.0,
            full_wall_s=2.0,
            outputs=[[11, 12], [21, 22]],
            mem={"peak_engine_delta_gib": 1.0},
        )

        self.assertEqual(
            row["generated_output_fingerprint_schema"],
            "wkvm.generated_output_token_ids.sha256.v1",
        )
        self.assertEqual(row["generated_output_request_ids"], ["bench-2-0", "bench-2-1"])
        self.assertEqual(row["generated_output_token_counts"], [2, 2])
        self.assertEqual(len(row["request_output_token_ids_sha256"]), 64)

    def test_vllm_capacity_uses_group_aware_cache_fields(self) -> None:
        cache_config = SimpleNamespace(
            kv_cache_size_tokens=131_072,
            kv_cache_max_concurrency=7.75,
            num_gpu_blocks=1,
            block_size=16,
        )
        llm = SimpleNamespace(
            llm_engine=SimpleNamespace(
                vllm_config=SimpleNamespace(cache_config=cache_config)
            )
        )

        telemetry = vllm_capacity_telemetry(llm, max_model_len=16_384)

        self.assertEqual(telemetry["kv_token_capacity"], 131_072)
        self.assertEqual(telemetry["kv_max_concurrency"], 7.75)
        self.assertFalse(telemetry["capacity_estimated"])
        self.assertEqual(
            telemetry["capacity_source"],
            "llm.llm_engine.vllm_config.cache_config",
        )

    def test_vllm_capacity_falls_back_to_blocks_and_model_length(self) -> None:
        cache_config = SimpleNamespace(
            num_gpu_blocks=8_192,
            block_size=16,
        )
        llm = SimpleNamespace(
            llm_engine=SimpleNamespace(cache_config=cache_config)
        )

        telemetry = vllm_capacity_telemetry(llm, max_model_len=16_384)

        self.assertEqual(telemetry["kv_token_capacity"], 131_072)
        self.assertEqual(telemetry["kv_max_concurrency"], 8.0)
        self.assertTrue(telemetry["capacity_estimated"])

    def test_vllm_runtime_sample_accepts_metric_objects_and_total_suffix(self) -> None:
        llm = SimpleNamespace(
            get_metrics=lambda: [
                SimpleNamespace(name="vllm:num_requests_running", value=3),
                SimpleNamespace(name="vllm:num_requests_running", value=2),
                SimpleNamespace(name="vllm:num_requests_waiting", value=4),
                SimpleNamespace(name="vllm:num_preemptions_total", value=7),
            ]
        )

        sample = vllm_runtime_telemetry_sample(llm)

        self.assertEqual(sample["running_requests"], 5.0)
        self.assertEqual(sample["waiting_requests"], 4.0)
        self.assertEqual(sample["preemptions_total"], 7.0)
        self.assertEqual(sample["source"], "llm.get_metrics")

    def test_residency_monitor_tracks_peaks_and_preemption_delta(self) -> None:
        samples = iter(
            [
                {
                    "source": "mock.metrics",
                    "running_requests": 0,
                    "waiting_requests": 0,
                    "preemptions_total": 11,
                },
                {
                    "source": "mock.metrics",
                    "running_requests": 16,
                    "waiting_requests": 3,
                    "preemptions_total": 13,
                },
                {
                    "source": "mock.metrics",
                    "running_requests": 0,
                    "waiting_requests": 0,
                    "preemptions_total": 13,
                },
            ]
        )
        monitor = ResidencyTelemetryMonitor(
            engine="vllm",
            capacity={
                "kv_token_capacity": 131_072,
                "kv_max_concurrency": 8.0,
                "capacity_source": "mock.cache",
            },
            sampler=lambda: next(samples),
            required_fields=(
                "kv_token_capacity",
                "kv_max_concurrency",
                "peak_running_requests",
                "peak_waiting_requests",
                "preemption_events",
            ),
            interval_s=60.0,
        )

        with monitor:
            monitor._sample_once(periodic=True)
        result = monitor.result()

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["peak_running_requests"], 16)
        self.assertEqual(result["peak_waiting_requests"], 3)
        self.assertEqual(result["preemption_events"], 2)
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["periodic_sample_count"], 1)
        self.assertEqual(result["active_periodic_sample_count"], 1)
        self.assertEqual(result["active_sample_count"], 1)
        self.assertEqual(result["sources"], ["mock.cache", "mock.metrics"])

    def test_residency_monitor_requires_active_sample_coverage(self) -> None:
        monitor = ResidencyTelemetryMonitor(
            engine="vllm",
            capacity={
                "kv_token_capacity": 131_072,
                "kv_max_concurrency": 8.0,
                "capacity_source": "mock.cache",
            },
            sampler=lambda: {
                "source": "mock.metrics",
                "running_requests": 0,
                "waiting_requests": 0,
                "preemptions_total": 0,
            },
            required_fields=(
                "kv_token_capacity",
                "kv_max_concurrency",
                "peak_running_requests",
                "peak_waiting_requests",
                "preemption_events",
            ),
            interval_s=60.0,
        )

        with monitor:
            pass
        result = monitor.result()

        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["active_sample_count"], 0)
        self.assertEqual(result["active_periodic_sample_count"], 0)
        self.assertIn(
            "active_periodic_sample_coverage",
            result["unavailable_fields"],
        )

    def test_residency_monitor_marks_missing_api_unavailable(self) -> None:
        def unavailable() -> dict[str, object]:
            raise TelemetryUnavailable("metrics not exposed")

        monitor = ResidencyTelemetryMonitor(
            engine="vllm",
            capacity={},
            sampler=unavailable,
            required_fields=("peak_running_requests", "preemption_events"),
        )

        with monitor:
            pass
        result = monitor.result()

        self.assertEqual(result["status"], "unavailable")
        self.assertFalse(result["available"])
        self.assertEqual(result["sample_count"], 0)
        self.assertEqual(result["error_count"], 1)
        self.assertEqual(
            result["unavailable_fields"],
            ["peak_running_requests", "preemption_events"],
        )

    def test_sglang_capacity_falls_back_to_internal_state(self) -> None:
        engine = SimpleNamespace(
            get_server_info=lambda: {
                "internal_states": [
                    {
                        "memory_usage": {"token_capacity": 65_536},
                        "effective_max_running_requests_per_dp": 32,
                    },
                    {
                        "memory_usage": {"token_capacity": 65_536},
                        "effective_max_running_requests_per_dp": 32,
                    },
                ]
            }
        )

        telemetry = sglang_capacity_telemetry(engine)

        self.assertEqual(telemetry["effective_token_capacity"], 131_072)
        self.assertEqual(telemetry["configured_max_running_requests"], 64)
        self.assertIn("internal_states", telemetry["capacity_source"])

    def test_sglang_runtime_sample_sums_load_snapshots(self) -> None:
        reader = SimpleNamespace(
            read_all=lambda: [
                SimpleNamespace(
                    num_running_reqs=5,
                    num_waiting_reqs=2,
                    num_used_tokens=20_000,
                ),
                {
                    "num_running_reqs": 7,
                    "num_queue_reqs": 3,
                    "num_used_tokens": 25_000,
                },
            ]
        )
        engine = SimpleNamespace(
            tokenizer_manager=SimpleNamespace(load_snapshot_reader=reader)
        )

        sample = sglang_runtime_telemetry_sample(engine)

        self.assertEqual(sample["running_requests"], 12.0)
        self.assertEqual(sample["waiting_requests"], 5.0)
        self.assertEqual(sample["used_tokens"], 45_000.0)

    def test_sglang_output_retractions_uses_per_request_metadata(self) -> None:
        outputs = [
            {"meta_info": {"num_retractions": 0}},
            {"meta_info": {"num_retractions": 2}},
            {"meta": {"retraction_count": 1}},
        ]

        retractions, captured = sglang_output_retractions(outputs, expected=3)

        self.assertEqual(retractions, 3)
        self.assertEqual(captured, 3)

    def test_residency_row_fields_separate_capacity_and_scheduler_limit(self) -> None:
        fields = residency_telemetry_row_fields(
            {
                "status": "complete",
                "kv_token_capacity": None,
                "effective_token_capacity": 90_000,
                "kv_max_concurrency": None,
                "configured_max_running_requests": 64,
                "peak_running_requests": 32,
                "peak_waiting_requests": 4,
                "peak_used_tokens": 80_000,
                "preemption_events": None,
                "output_retractions": 2,
            },
            tokens_per_request=18_000,
        )

        self.assertEqual(fields["token_capacity"], 90_000)
        self.assertIsNone(fields["kv_max_concurrency"])
        self.assertEqual(fields["configured_max_running_requests"], 64)
        self.assertEqual(fields["full_length_context_capacity"], 5.0)
        self.assertEqual(fields["max_running"], 32)
        self.assertEqual(fields["max_waiting"], 4)
        self.assertNotIn("max_resident_state_slots", fields)
        self.assertEqual(fields["retraction_events"], 2)

    def test_green_requires_full_success_and_memory_gate(self) -> None:
        args = SimpleNamespace(mem_cap_gib=19.0, headroom_gib=1.0)
        complete = {
            "B": 32,
            "success_count": 32,
            "error_count": 0,
            "error": None,
            "peak_engine_delta_gib": 17.5,
        }

        self.assertTrue(row_green(complete, args))
        self.assertFalse(row_green({**complete, "success_count": 0}, args))
        self.assertFalse(row_green({**complete, "peak_engine_delta_gib": 18.1}, args))

    def test_sglang_language_model_override_promotes_text_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir)
            (model_path / "config.json").write_text(
                json.dumps(
                    {
                        "architectures": ["Gemma4ForConditionalGeneration"],
                        "vision_config": {"hidden_size": 1152},
                        "text_config": {
                            "model_type": "gemma4_text",
                            "head_dim": 256,
                            "global_head_dim": 512,
                            "num_key_value_heads": 2,
                            "num_global_key_value_heads": 4,
                        },
                    }
                )
            )

            override = sglang_language_model_override(str(model_path))

        self.assertEqual(override["architectures"], ["Gemma4ForCausalLM"])
        self.assertEqual(override["model_type"], "gemma4_text")
        self.assertEqual(override["head_dim"], 512)
        self.assertEqual(override["global_head_dim"], 512)
        self.assertEqual(override["swa_head_dim"], 256)
        self.assertEqual(override["swa_v_head_dim"], 256)
        self.assertEqual(override["num_key_value_heads"], 4)
        self.assertEqual(override["swa_num_key_value_heads"], 2)
        self.assertNotIn("vision_config", override)

    def test_sglang_server_launcher_forces_text_only_mode(self) -> None:
        args = build_sglang_server_arg_parser().parse_args(
            ["--model-path", "/models/gemma-4-E4B-it"]
        )
        with mock.patch(
            "experiments.sglang_gemma_server.sglang_language_model_override",
            return_value={"architectures": ["Gemma4ForCausalLM"]},
        ):
            kwargs = sglang_server_kwargs(args)

        self.assertFalse(kwargs["enable_multimodal"])
        self.assertEqual(kwargs["context_length"], 15_232)
        self.assertEqual(kwargs["max_running_requests"], 32)
        self.assertEqual(kwargs["sampling_defaults"], "openai")
        self.assertEqual(
            json.loads(kwargs["json_model_override_args"])["architectures"],
            ["Gemma4ForCausalLM"],
        )


if __name__ == "__main__":
    unittest.main()
