import json
import tempfile
import unittest
from pathlib import Path

from experiments.benchmark_contract import (
    ComparisonContractError,
    validate_comparable,
)
from experiments import gemma_bench_report


class TestBenchmarkContract(unittest.TestCase):
    def direct_payload(
        self,
        engine: str,
        *,
        batches: tuple[int, ...] = (2,),
        prompt_lengths: dict[int, list[int]] | None = None,
        schema: str | None = None,
        dtype: str = "bfloat16",
        git_commit: str = "a" * 40,
        model_path: str = "/models/gemma-4-e4b-it",
        memory_kind: str = "peak_reserved_gib",
    ) -> dict:
        if schema is None:
            schema = (
                "wkvm.native_gemma_bench.v1"
                if engine == "wkvm-native"
                else "wkvm.incumbent_gemma_bench.v1"
            )
        rows = []
        for batch in batches:
            lengths = (
                prompt_lengths[batch]
                if prompt_lengths is not None and batch in prompt_lengths
                else [512] * batch
            )
            rows.append(
                {
                    "B": batch,
                    "success_count": batch,
                    "error_count": 0,
                    "prompt_lengths": lengths,
                    "agg_decode_tok_s": 100.0 + batch,
                    "green": True,
                    memory_kind: 12.0,
                }
            )
        payload = {
            "schema": schema,
            "engine": engine,
            "context_tokens_per_session": 512,
            "decode_tokens_per_session": 8,
            "prompt_lengths_mode": "uniform",
            "dtype": dtype,
            "model_path": model_path,
            "git_commit": git_commit,
            "launch_command": f"python bench.py --engine {engine}",
            "mem_cap_gib": 19.0,
            "headroom_gib": 1.0,
            "rows": rows,
        }
        if engine == "vllm":
            payload["engine_config"] = {"vllm_version": "0.24.0"}
        elif engine == "sglang":
            payload["engine_config"] = {"sglang_version": "0.5.14"}
        return payload

    def serving_payload(
        self,
        engine: str,
        *,
        backend: str = "openai-completions",
        request_count: int = 2,
    ) -> dict:
        version = {
            "vllm-http-stream": "0.24.0",
            "sglang-http-stream": "0.5.14",
        }.get(engine, "0.0.1")
        return {
            "schema": "wkvm.serving_bench.v1",
            "engine": engine,
            "backend": backend,
            "context_tokens_per_session": 512,
            "decode_tokens_per_session": 8,
            "prompt_lengths_mode": "uniform",
            "model_path": "/models/gemma-4-e4b-it",
            "served_model": "gemma-4-e4b-it",
            "semantics": "full_kv",
            "sampling": {
                "temperature": 0.0,
                "ignore_eos": True,
                "stream": True,
                "max_tokens": 8,
            },
            "git_commit": "a" * 40,
            "launch_command": f"python serve_bench.py --engine {engine}",
            "requests_per_row": request_count,
            "prompt_token_source": "synthetic",
            "prompt_reuse_policy": "disjoint_across_measured_and_warmup_rows",
            "warmup_requests": 1,
            "warmup_output_tokens": 4,
            "warmup_row_offset": 64,
            "extra_body": None,
            "provenance": {
                "schema": "wkvm.serving_bench.provenance.v2",
                "engine": {
                    "label": engine,
                    "version": version,
                    "version_source": "server_environment",
                },
                "target_server": {
                    "launch_command": f"serve {engine} --port 8000",
                    "launch_command_source": "operator_supplied",
                    "config": {"test_fixture": True},
                    "config_source": "operator_supplied",
                },
                "gpu": {
                    "index": 0,
                    "uuid": "GPU-test",
                    "name": "Test GPU",
                    "driver_version": "1.2.3",
                    "memory_total_mib": 24000,
                },
                "gpu_memory_monitor": {
                    "enabled": False,
                    "scope": None,
                    "source": None,
                    "device_selector": None,
                    "sample_interval_s": None,
                },
            },
            "rows": [
                {
                    "B": 2,
                    "request_count": request_count,
                    "success_count": request_count,
                    "error_count": 0,
                    "prompt_lengths": [512] * request_count,
                    "request_output_tok_s": 80.0,
                    "p50_ttft_s": 0.1,
                    "p95_ttft_s": 0.2,
                    "p50_itl_s": 0.01,
                    "p95_itl_s": 0.02,
                    "itl_valid_request_count": request_count,
                    "itl_sample_count": request_count * 7,
                    "output_token_count_exact_requests": request_count,
                    "output_token_count_sources": ["usage"],
                }
            ],
        }

    def artifacts(self, *payloads: dict):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        artifacts = []
        for index, payload in enumerate(payloads):
            path = root / f"result-{index}.json"
            path.write_text(json.dumps(payload))
            artifacts.append((path, payload))
        return tmp, artifacts

    def test_accepts_full_ladder_and_warns_on_memory_method(self) -> None:
        native = self.direct_payload("wkvm-native", batches=(1, 2, 4))
        vllm = self.direct_payload(
            "vllm",
            batches=(1, 2, 4),
            git_commit="b" * 40,
            memory_kind="peak_engine_delta_gib",
        )
        tmp, artifacts = self.artifacts(native, vllm)
        with tmp:
            result = validate_comparable(artifacts)

        self.assertEqual(result.benchmark_path, "direct-offline")
        self.assertEqual(result.paired_batch_sizes, (1, 2, 4))
        self.assertIn("benchmark harness commits differ across inputs", result.warnings)
        self.assertIn(
            "memory metric kinds differ; green/memory values are not cross-engine comparable",
            result.warnings,
        )

    def test_rejects_mixed_direct_and_http_paths(self) -> None:
        direct = self.direct_payload("wkvm-native")
        serving = self.serving_payload("wkvm-native-openai-completions")
        tmp, artifacts = self.artifacts(direct, serving)
        with tmp, self.assertRaisesRegex(ComparisonContractError, "mix benchmark paths"):
            validate_comparable(artifacts)

    def test_rejects_different_serving_backends(self) -> None:
        wkvm = self.serving_payload("wkvm-native-openai-completions")
        other = self.serving_payload("other-engine", backend="wkvm")
        tmp, artifacts = self.artifacts(wkvm, other)
        with tmp, self.assertRaisesRegex(ComparisonContractError, "mix benchmark paths"):
            validate_comparable(artifacts)

    def test_serving_uses_request_count_for_success_and_workload_pairing(self) -> None:
        wkvm = self.serving_payload(
            "wkvm-native-openai-completions", request_count=6
        )
        other = self.serving_payload("other-engine", request_count=6)
        tmp, artifacts = self.artifacts(wkvm, other)
        with tmp:
            result = validate_comparable(artifacts)
        self.assertEqual(result.paired_batch_sizes, (2,))

        other = self.serving_payload("other-engine", request_count=4)
        other["requests_per_row"] = 6
        tmp, artifacts = self.artifacts(wkvm, other)
        with tmp, self.assertRaisesRegex(ComparisonContractError, "request_count"):
            validate_comparable(artifacts)

    def test_serving_missing_engine_version_is_a_warning(self) -> None:
        wkvm = self.serving_payload("wkvm-native-openai-completions")
        vllm = self.serving_payload("vllm-http-stream")
        vllm["provenance"]["engine"]["version"] = None
        tmp, artifacts = self.artifacts(wkvm, vllm)
        with tmp:
            result = validate_comparable(artifacts)
        self.assertIn(
            "server engine versions are missing for vllm-http-stream",
            result.warnings,
        )

    def test_serving_keeps_legacy_artifacts_comparable_with_warning(self) -> None:
        wkvm = self.serving_payload("wkvm-native-openai-completions")
        vllm = self.serving_payload("vllm-http-stream")
        wkvm.pop("provenance")
        vllm.pop("provenance")
        vllm["engine_config"] = {"vllm_version": "0.24.0"}
        tmp, artifacts = self.artifacts(wkvm, vllm)
        with tmp:
            result = validate_comparable(artifacts)
        self.assertIn(
            "structured serving provenance is missing for "
            "vllm-http-stream, wkvm-native-openai-completions",
            result.warnings,
        )

    def test_serving_keeps_v1_provenance_comparable_with_warning(self) -> None:
        wkvm = self.serving_payload("wkvm-native-openai-completions")
        vllm = self.serving_payload("vllm-http-stream")
        for payload in (wkvm, vllm):
            payload["provenance"]["schema"] = "wkvm.serving_bench.provenance.v1"
            payload["provenance"].pop("target_server")
        tmp, artifacts = self.artifacts(wkvm, vllm)
        with tmp:
            result = validate_comparable(artifacts)
        self.assertIn(
            "legacy structured serving provenance does not prove target server "
            "launch commands for vllm-http-stream, wkvm-native-openai-completions",
            result.warnings,
        )

    def test_serving_v2_requires_operator_target_launch_provenance(self) -> None:
        wkvm = self.serving_payload("wkvm-native-openai-completions")
        vllm = self.serving_payload("vllm-http-stream")
        vllm["provenance"]["target_server"]["launch_command"] = None
        tmp, artifacts = self.artifacts(wkvm, vllm)
        with tmp, self.assertRaisesRegex(
            ComparisonContractError, "target server launch provenance"
        ):
            validate_comparable(artifacts)

    def test_serving_rejects_mismatched_gpu_monitoring_policy(self) -> None:
        wkvm = self.serving_payload("wkvm-native-openai-completions")
        vllm = self.serving_payload("vllm-http-stream")
        vllm["provenance"]["gpu_memory_monitor"] = {
            "enabled": True,
            "scope": "whole_device",
            "source": "nvidia-smi",
            "device_selector": "0",
            "sample_interval_s": 0.1,
        }
        tmp, artifacts = self.artifacts(wkvm, vllm)
        with tmp, self.assertRaisesRegex(
            ComparisonContractError, "GPU-monitoring policies"
        ):
            validate_comparable(artifacts)

    def test_serving_warns_on_driver_difference(self) -> None:
        wkvm = self.serving_payload("wkvm-native-openai-completions")
        vllm = self.serving_payload("vllm-http-stream")
        vllm["provenance"]["gpu"]["driver_version"] = "9.9.9"
        tmp, artifacts = self.artifacts(wkvm, vllm)
        with tmp:
            result = validate_comparable(artifacts)
        self.assertIn("GPU driver versions differ across inputs", result.warnings)

    def test_serving_warns_on_partial_whole_gpu_memory(self) -> None:
        wkvm = self.serving_payload("wkvm-native-openai-completions")
        vllm = self.serving_payload("vllm-http-stream")
        for payload in (wkvm, vllm):
            payload["provenance"]["gpu_memory_monitor"] = {
                "enabled": True,
                "scope": "whole_device",
                "source": "nvidia-smi",
                "device_selector": "0",
                "sample_interval_s": 0.1,
            }
        wkvm["rows"][0]["gpu_memory"] = {
            "schema": "wkvm.whole_gpu_memory.v1",
            "scope": "whole_device",
            "source": "nvidia-smi",
            "peak_used_mib": 12000,
        }
        tmp, artifacts = self.artifacts(wkvm, vllm)
        with tmp:
            result = validate_comparable(artifacts)
        self.assertTrue(
            any(
                warning.startswith("whole-GPU memory is missing")
                for warning in result.warnings
            )
        )

    def test_rejects_no_shared_batch_size(self) -> None:
        native = self.direct_payload("wkvm-native", batches=(1,))
        vllm = self.direct_payload("vllm", batches=(2,))
        tmp, artifacts = self.artifacts(native, vllm)
        with tmp, self.assertRaisesRegex(ComparisonContractError, "concurrency ladders"):
            validate_comparable(artifacts)

    def test_rejects_partial_concurrency_ladder(self) -> None:
        native = self.direct_payload("wkvm-native", batches=(1, 2, 4))
        vllm = self.direct_payload("vllm", batches=(2,))
        tmp, artifacts = self.artifacts(native, vllm)
        with tmp, self.assertRaisesRegex(ComparisonContractError, "concurrency ladders"):
            validate_comparable(artifacts)

    def test_serving_requires_exact_counts_and_equal_policy(self) -> None:
        wkvm = self.serving_payload("wkvm-native-openai-completions")
        other = self.serving_payload("other-engine")
        other["rows"][0]["output_token_count_exact_requests"] = 1
        tmp, artifacts = self.artifacts(wkvm, other)
        with tmp, self.assertRaisesRegex(ComparisonContractError, "exact output-token"):
            validate_comparable(artifacts)

        other = self.serving_payload("other-engine")
        other["served_model"] = "different-model"
        tmp, artifacts = self.artifacts(wkvm, other)
        with tmp, self.assertRaisesRegex(ComparisonContractError, "different model"):
            validate_comparable(artifacts)

    def test_serving_labels_semantic_mismatch(self) -> None:
        wkvm = self.serving_payload("wkvm-native-openai-completions")
        wkvm["semantics"] = "routed_span_approximate"
        vllm = self.serving_payload("vllm-http-stream")
        tmp, artifacts = self.artifacts(wkvm, vllm)
        with tmp:
            result = validate_comparable(artifacts)
        self.assertIn(
            "semantic modes differ (full_kv, routed_span_approximate); "
            "throughput is not quality-equivalent",
            result.warnings,
        )

    def test_rejects_shape_mismatch_and_missing_throughput(self) -> None:
        native = self.direct_payload("wkvm-native")
        vllm = self.direct_payload("vllm")
        vllm["context_tokens_per_session"] = 1024
        tmp, artifacts = self.artifacts(native, vllm)
        with tmp, self.assertRaisesRegex(ComparisonContractError, "different ctx/out"):
            validate_comparable(artifacts)

        vllm = self.direct_payload("vllm")
        del vllm["rows"][0]["agg_decode_tok_s"]
        tmp, artifacts = self.artifacts(native, vllm)
        with tmp, self.assertRaisesRegex(ComparisonContractError, "agg_decode_tok_s"):
            validate_comparable(artifacts)

    def test_rejects_different_prompt_lengths_at_paired_batch(self) -> None:
        native = self.direct_payload("wkvm-native")
        vllm = self.direct_payload("vllm", prompt_lengths={2: [512, 511]})
        tmp, artifacts = self.artifacts(native, vllm)
        with tmp, self.assertRaisesRegex(ComparisonContractError, "different prompt_lengths"):
            validate_comparable(artifacts)

    def test_rejects_failed_or_partial_paired_row(self) -> None:
        native = self.direct_payload("wkvm-native")
        vllm = self.direct_payload("vllm")
        vllm["rows"][0]["success_count"] = 1
        vllm["rows"][0]["error_count"] = 1
        tmp, artifacts = self.artifacts(native, vllm)
        with tmp, self.assertRaisesRegex(ComparisonContractError, "not fully successful"):
            validate_comparable(artifacts)

    def test_rejects_dtype_and_memory_budget_mismatches(self) -> None:
        native = self.direct_payload("wkvm-native")
        vllm = self.direct_payload("vllm", dtype="float16")
        tmp, artifacts = self.artifacts(native, vllm)
        with tmp, self.assertRaisesRegex(ComparisonContractError, "different dtypes"):
            validate_comparable(artifacts)

        vllm = self.direct_payload("vllm")
        vllm["headroom_gib"] = 2.0
        tmp, artifacts = self.artifacts(native, vllm)
        with tmp, self.assertRaisesRegex(ComparisonContractError, "caps/headroom"):
            validate_comparable(artifacts)

    def test_rejects_missing_incumbent_version(self) -> None:
        native = self.direct_payload("wkvm-native")
        vllm = self.direct_payload("vllm")
        del vllm["engine_config"]
        tmp, artifacts = self.artifacts(native, vllm)
        with tmp, self.assertRaisesRegex(ComparisonContractError, "vllm_version"):
            validate_comparable(artifacts)

    def test_report_integration_is_opt_in(self) -> None:
        native = self.direct_payload("wkvm-native")
        vllm = self.direct_payload("vllm")
        tmp, artifacts = self.artifacts(native, vllm)
        with tmp:
            paths = [path for path, _payload in artifacts]
            normal = gemma_bench_report.render(paths)
            strict = gemma_bench_report.render(paths, require_comparable=True)

        self.assertNotIn("Comparison contract:", normal)
        self.assertIn("Comparison contract: **PASS (Stage 1 workload)**", strict)
        self.assertIn("paired `B=2`", strict)

    def test_report_renders_serving_itl_validity(self) -> None:
        wkvm = self.serving_payload(
            "wkvm-native-openai-completions", request_count=6
        )
        other = self.serving_payload("other-engine", request_count=6)
        tmp, artifacts = self.artifacts(wkvm, other)
        with tmp:
            paths = [path for path, _payload in artifacts]
            text = gemma_bench_report.render(paths, require_comparable=True)

        self.assertIn("| ITL exact |", text)
        self.assertIn("6/6 req; 42 samples", text)
        self.assertIn("ttft 0.100/0.200", text)
        self.assertIn("p50/p95 0.010/0.020", text)
        self.assertIn("count 6/6 exact (usage)", text)
        self.assertIn("| full_kv |", text)


if __name__ == "__main__":
    unittest.main()
