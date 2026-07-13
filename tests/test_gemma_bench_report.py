import json
import tempfile
import unittest
from pathlib import Path

from experiments import gemma_bench_report


class TestGemmaBenchReport(unittest.TestCase):
    def write_payload(self, tmp: Path, name: str, payload: dict) -> Path:
        path = tmp / name
        path.write_text(json.dumps(payload))
        return path

    def native_payload(
        self,
        *,
        ctx: int = 512,
        out: int = 8,
        prompt_mode: str = "uniform",
        row: dict | None = None,
        prompt_hash: str | None = "a" * 64,
    ) -> dict:
        native_row = {
            "B": 2,
            "success_count": 2,
            "green": True,
            "agg_decode_tok_s": 13.101,
            "peak_reserved_gib": 14.301,
            "model_forward_backend": "wkvm_native_gemma_forward_bridge",
            "uses_hf_transformer_forward": False,
            "uses_hf_model_construction": False,
            "native_gemma_checkpoint_loader": True,
        }
        if prompt_hash is not None:
            native_row["prompt_fingerprint"] = {
                "schema": "wkvm.prompt_token_ids.sha256.v1",
                "prompt_token_source": "synthetic",
                "prompt_count": 2,
                "prompt_total_tokens": 1024,
                "prompt_lengths": [512, 512],
                "prompt_token_ids_sha256": prompt_hash,
            }
        if row is not None:
            native_row = row
        return {
            "schema": "wkvm.native_gemma_bench.v1",
            "engine": "wkvm-native",
            "config": {"decode_microbatch_rows": 16},
            "context_tokens_per_session": ctx,
            "decode_tokens_per_session": out,
            "prompt_lengths_mode": prompt_mode,
            "uses_hf_tokenizer": False,
            "uses_hf_config": False,
            "native_gemma_config_loader": True,
            "native_no_hf_requirement": {
                "checked_successful_rows": 1,
                "passed": True,
                "required": True,
                "violations": [],
            },
            "rows": [native_row],
        }

    def incumbent_payload(
        self,
        *,
        engine: str = "vllm",
        ctx: int = 512,
        out: int = 8,
        prompt_mode: str = "uniform",
        prompt_hash: str | None = None,
    ) -> dict:
        row = {
            "B": 2,
            "success_count": 2,
            "green": True,
            "agg_decode_tok_s": 21.5,
            "peak_engine_delta_gib": 12.0,
        }
        if prompt_hash is not None:
            row["prompt_fingerprint"] = {
                "schema": "wkvm.prompt_token_ids.sha256.v1",
                "prompt_token_source": "synthetic",
                "prompt_count": 2,
                "prompt_total_tokens": 1024,
                "prompt_lengths": [512, 512],
                "prompt_token_ids_sha256": prompt_hash,
            }
        return {
            "schema": "wkvm.incumbent_gemma_bench.v1",
            "engine": engine,
            "context_tokens_per_session": ctx,
            "decode_tokens_per_session": out,
            "prompt_lengths_mode": prompt_mode,
            "rows": [row],
        }

    def test_render_exposes_native_no_hf_columns(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            native = self.write_payload(tmp, "native.json", self.native_payload())
            incumbent = self.write_payload(
                tmp, "vllm.json", self.incumbent_payload()
            )

            text = gemma_bench_report.render(
                [native, incumbent],
                require_same_shape=True,
                require_native_no_hf=True,
            )

        self.assertIn(
            "forward backend | HF fwd | HF construct | HF tok | HF cfg | native cfg | native ckpt",
            text,
        )
        self.assertIn("prompt fingerprint", text)
        self.assertIn("synthetic aaaaaaaaaaaa (2 prompts / 1024 tok)", text)
        self.assertIn(
            "wkvm_native_gemma_forward_bridge | no | no | no | no | yes | yes",
            text,
        )
        self.assertIn("pass (1 rows, required)", text)
        self.assertIn("| vllm | ctx=512 out=8 prompt=uniform", text)
        self.assertIn("- | - | - | - | - | - | n/a", text)

    def test_render_exposes_serving_provenance_and_whole_gpu_caveat(self) -> None:
        payload = {
            "schema": "wkvm.serving_bench.v1",
            "engine": "vllm-http-stream",
            "launch_command": "python experiments/wkvm_serving_bench.py --engine vllm-http-stream",
            "context_tokens_per_session": 512,
            "decode_tokens_per_session": 8,
            "prompt_lengths_mode": "uniform",
            "provenance": {
                "schema": "wkvm.serving_bench.provenance.v2",
                "engine": {
                    "label": "vllm-http-stream",
                    "version": "0.24.0",
                    "version_source": "server_environment",
                },
                "target_server": {
                    "launch_command": "vllm serve /models/gemma --port 8001",
                    "launch_command_source": "operator_supplied",
                    "config": {
                        "max_model_len": 13824,
                        "tensor_parallel_size": 1,
                    },
                    "config_source": "operator_supplied",
                },
                "client_environment": {
                    "python_version": "3.12.1",
                    "packages": {"wkvm": "0.0.1", "torch": "2.10.0"},
                },
                "gpu": {
                    "index": 0,
                    "uuid": "GPU-test",
                    "name": "Test GPU",
                    "driver_version": "595.71.05",
                    "memory_total_mib": 24576,
                },
            },
            "rows": [
                {
                    "B": 2,
                    "request_count": 2,
                    "success_count": 2,
                    "request_output_tok_s": 100.0,
                    "gpu_memory": {
                        "schema": "wkvm.whole_gpu_memory.v1",
                        "scope": "whole_device",
                        "source": "nvidia-smi",
                        "sample_count": 4,
                        "baseline_used_mib": 12000,
                        "peak_used_mib": 12544,
                        "peak_delta_mib": 544,
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            result = self.write_payload(tmp, "serving.json", payload)
            text = gemma_bench_report.render([result])

        self.assertIn("## Environment Provenance", text)
        self.assertIn("0.24.0 | server_environment", text)
        self.assertIn("Test GPU (index 0) | 595.71.05 | 24.000 GiB", text)
        self.assertIn("Python 3.12.1; wkvm 0.0.1; torch 2.10.0", text)
        self.assertIn("12.250 GiB (baseline 11.719 GiB", text)
        self.assertIn("whole GPU peak", text)
        self.assertIn("includes every process", text)
        self.assertIn("## Launch Provenance", text)
        self.assertIn("vllm serve /models/gemma --port 8001", text)
        self.assertIn(
            '{"max_model_len":13824,"tensor_parallel_size":1}', text
        )
        self.assertIn(
            "python experiments/wkvm_serving_bench.py --engine vllm-http-stream",
            text,
        )

    def test_require_same_shape_rejects_mixed_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            native = self.write_payload(tmp, "native.json", self.native_payload())
            incumbent = self.write_payload(
                tmp, "vllm.json", self.incumbent_payload(ctx=1024)
            )

            with self.assertRaisesRegex(ValueError, "same-shape requirement failed"):
                gemma_bench_report.render(
                    [native, incumbent],
                    require_same_shape=True,
                )

    def test_require_same_prompt_fingerprint_accepts_matching_rows(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            native = self.write_payload(tmp, "native.json", self.native_payload())
            incumbent = self.write_payload(
                tmp,
                "vllm.json",
                self.incumbent_payload(prompt_hash="a" * 64),
            )

            text = gemma_bench_report.render(
                [native, incumbent],
                require_same_shape=True,
                require_same_prompt_fingerprint=True,
            )

        self.assertIn("synthetic aaaaaaaaaaaa", text)

    def test_require_same_prompt_fingerprint_rejects_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            native = self.write_payload(tmp, "native.json", self.native_payload())
            incumbent = self.write_payload(
                tmp,
                "vllm.json",
                self.incumbent_payload(prompt_hash="b" * 64),
            )

            with self.assertRaisesRegex(
                ValueError,
                "prompt fingerprints differ",
            ):
                gemma_bench_report.render(
                    [native, incumbent],
                    require_same_shape=True,
                    require_same_prompt_fingerprint=True,
                )

    def test_require_same_prompt_fingerprint_rejects_missing_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            native = self.write_payload(tmp, "native.json", self.native_payload())
            incumbent = self.write_payload(tmp, "vllm.json", self.incumbent_payload())

            with self.assertRaisesRegex(
                ValueError,
                "missing prompt fingerprint",
            ):
                gemma_bench_report.render(
                    [native, incumbent],
                    require_same_shape=True,
                    require_same_prompt_fingerprint=True,
                )

    def test_require_same_prompt_fingerprint_rejects_rowless_payload(self) -> None:
        payload = self.native_payload()
        payload["rows"] = []
        payload["fatal_error"] = {
            "type": "OutOfMemoryError",
            "phase": "model_load",
            "message": "CUDA out of memory",
        }
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            native = self.write_payload(tmp, "native.json", payload)

            with self.assertRaisesRegex(
                ValueError,
                "no benchmark rows",
            ):
                gemma_bench_report.render(
                    [native],
                    require_same_prompt_fingerprint=True,
                )

    def test_require_native_no_hf_rejects_missing_row_evidence(self) -> None:
        stale_native_row = {
            "B": 2,
            "success_count": 2,
            "green": True,
            "agg_decode_tok_s": 13.101,
        }
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            native = self.write_payload(
                tmp, "native.json", self.native_payload(row=stale_native_row)
            )

            with self.assertRaisesRegex(
                ValueError, "uses_hf_transformer_forward_not_false"
            ):
                gemma_bench_report.render(
                    [native],
                    require_native_no_hf=True,
                )

    def test_require_native_no_hf_rejects_setup_boundary_violations(self) -> None:
        payload = self.native_payload()
        payload["uses_hf_tokenizer"] = True
        payload["uses_hf_config"] = True
        payload["native_gemma_config_loader"] = False
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            native = self.write_payload(tmp, "native.json", payload)

            with self.assertRaisesRegex(
                ValueError, "uses_hf_tokenizer_not_false"
            ):
                gemma_bench_report.render(
                    [native],
                    require_native_no_hf=True,
                )

    def test_require_native_no_hf_rejects_missing_setup_evidence(self) -> None:
        payload = self.native_payload()
        payload.pop("uses_hf_tokenizer")
        payload.pop("uses_hf_config")
        payload.pop("native_gemma_config_loader")
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            native = self.write_payload(tmp, "native.json", payload)

            with self.assertRaisesRegex(
                ValueError, "native_gemma_config_loader_not_true"
            ):
                gemma_bench_report.render(
                    [native],
                    require_native_no_hf=True,
                )

    def test_require_native_no_hf_skips_failed_native_rows(self) -> None:
        payload = self.native_payload()
        payload["rows"].append(
            {
                "B": 16,
                "success_count": 0,
                "green": False,
                "error": "CUDA out of memory",
            }
        )
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            native = self.write_payload(tmp, "native.json", payload)

            text = gemma_bench_report.render(
                [native],
                require_native_no_hf=True,
            )

        self.assertIn("CUDA out of memory", text)

    def test_render_shows_rowless_native_setup_failure(self) -> None:
        payload = self.native_payload()
        payload["rows"] = []
        payload["fatal_error"] = {
            "type": "OutOfMemoryError",
            "phase": "model_load",
            "message": "CUDA out of memory",
        }
        payload["native_no_hf_requirement"] = {
            "checked_successful_rows": 0,
            "passed": False,
            "required": True,
            "violations": [
                {"B": None, "problems": ["no_successful_rows_to_check"]}
            ],
        }
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            native = self.write_payload(tmp, "native.json", payload)

            text = gemma_bench_report.render([native])

        self.assertIn("fail (0 rows, required)", text)
        self.assertIn("model_load: OutOfMemoryError: CUDA out of memory", text)
        self.assertIn("| wkvm-native row-cap 16 | ctx=512 out=8 prompt=uniform", text)
        self.assertIn(" | - | 0/- | no | ", text)

    def test_require_native_no_hf_requires_a_native_payload(self) -> None:
        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp = Path(raw_tmp)
            incumbent = self.write_payload(
                tmp, "vllm.json", self.incumbent_payload()
            )

            with self.assertRaisesRegex(ValueError, "no_wkvm_native_payloads"):
                gemma_bench_report.render(
                    [incumbent],
                    require_native_no_hf=True,
                )


if __name__ == "__main__":
    unittest.main()
