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
        if row is not None:
            native_row = row
        return {
            "schema": "wkvm.native_gemma_bench.v1",
            "engine": "wkvm-native",
            "config": {"decode_microbatch_rows": 16},
            "context_tokens_per_session": ctx,
            "decode_tokens_per_session": out,
            "prompt_lengths_mode": prompt_mode,
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
    ) -> dict:
        return {
            "schema": "wkvm.incumbent_gemma_bench.v1",
            "engine": engine,
            "context_tokens_per_session": ctx,
            "decode_tokens_per_session": out,
            "prompt_lengths_mode": prompt_mode,
            "rows": [
                {
                    "B": 2,
                    "success_count": 2,
                    "green": True,
                    "agg_decode_tok_s": 21.5,
                    "peak_engine_delta_gib": 12.0,
                }
            ],
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

        self.assertIn("forward backend | HF fwd | HF construct | native ckpt", text)
        self.assertIn("wkvm_native_gemma_forward_bridge | no | no | yes", text)
        self.assertIn("pass (1 rows, required)", text)
        self.assertIn("| vllm | ctx=512 out=8 prompt=uniform", text)
        self.assertIn("- | - | - | n/a", text)

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
