from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments.incumbent_gemma_bench import (
    VramMonitor,
    make_row,
    measure_vllm_generation,
    sglang_language_model_override,
    vllm_request_metrics_timing,
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


if __name__ == "__main__":
    unittest.main()
