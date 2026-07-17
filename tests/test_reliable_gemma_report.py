from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from experiments.benchmark_identity import (
    model_checkpoint_identity,
    source_worktree_identity,
)
from experiments.reliable_gemma_report import (
    ReliableReportError,
    build_arg_parser,
    build_summary,
    render_markdown,
    write_report,
)


COMMIT = "a" * 40
TREE = "b" * 40
PROMPT_DIGEST = "c" * 64
OUTPUT_DIGEST = "d" * 64
GPU_MODEL = "NVIDIA A800-SXM4-80GB"
DRIVER = "570.124.06"
GPU_INDEX = 3
GPU_UUID = "GPU-test-fixed"
SOURCE_EXCLUDED_PATH_PATTERNS = [
    "experiments/results/**",
    "**/__pycache__/**",
    ".pytest_cache/**",
    "**/*.egg-info/**",
    ".venv/**",
    "build/**",
    "dist/**",
]
SOURCE_IDENTITY_SCOPE = (
    "git tracked and all untracked worktree files excluding declared generated artifacts"
)


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


SOURCE_IDENTITY_FIELDS = {
    "git_commit": COMMIT,
    "git_head_tree": TREE,
    "git_status_sha256": hashlib.sha256(b"").hexdigest(),
    "git_tracked_diff_sha256": hashlib.sha256(b"").hexdigest(),
    "worktree_manifest_sha256": "1" * 64,
}
SOURCE_IDENTITY_SHA256 = canonical_sha256(SOURCE_IDENTITY_FIELDS)
SOURCE_IDENTITY = {
    "schema": "wkvm.git_worktree_identity.sha256.v1",
    "repo_root": "/repo/wkvm",
    "scope": SOURCE_IDENTITY_SCOPE,
    "excluded_path_patterns": SOURCE_EXCLUDED_PATH_PATTERNS,
    "excluded_paths": [],
    "git_worktree_dirty": False,
    "tracked_file_count": 200,
    "untracked_file_count": 0,
    "worktree_file_count": 200,
    **SOURCE_IDENTITY_FIELDS,
    "identity_sha256": SOURCE_IDENTITY_SHA256,
    "error": None,
}
MODEL_FILES = [
    {"path": "config.json", "size_bytes": 100, "sha256": "2" * 64},
    {"path": "model.safetensors", "size_bytes": 1_000, "sha256": "3" * 64},
    {"path": "tokenizer.json", "size_bytes": 200, "sha256": "4" * 64},
]
MODEL_MANIFEST_SHA256 = canonical_sha256(MODEL_FILES)
MODEL_IDENTITY = {
    "schema": "wkvm.model_checkpoint_identity.sha256.v1",
    "model_root": "/models/gemma-4-E4B-it",
    "excluded_path_patterns": [".cache/**"],
    "file_count": len(MODEL_FILES),
    "total_bytes": sum(entry["size_bytes"] for entry in MODEL_FILES),
    "files": MODEL_FILES,
    "manifest_sha256": MODEL_MANIFEST_SHA256,
    "error": None,
}


class ReliableGemmaReportTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def payload(self, engine: str, repeat: int, *, batch: int = 2) -> dict:
        output_tokens = 4
        prompt_lengths = [10] * batch
        prompt_total = sum(prompt_lengths)
        batch_wall = 4.0 + repeat
        cohort_wall = 2.0 + repeat * 0.1
        decode_interval = 3.0 + repeat * 0.1
        peak_used = 20_000 + repeat * 100
        baseline = 1_000
        device_index = GPU_INDEX
        device_selector = str(device_index)
        device_uuid = GPU_UUID
        memory = {
            "schema": "wkvm.whole_gpu_memory.v1",
            "scope": "whole_device",
            "source": "nvidia-smi",
            "device_selector": device_selector,
            "device_index": device_index,
            "device_uuid": device_uuid,
            "gpu_name": GPU_MODEL,
            "driver_version": DRIVER,
            "memory_total_mib": 81_920,
            "sample_interval_s": 0.1,
            "sample_count": 20,
            "baseline_used_mib": baseline,
            "peak_used_mib": peak_used,
            "peak_delta_mib": peak_used - baseline,
            "query_error_count": 0,
            "error": None,
        }
        prompt_fingerprint = {
            "schema": "wkvm.prompt_token_ids.sha256.v1",
            "prompt_token_source": "synthetic",
            "prompt_count": batch,
            "prompt_total_tokens": prompt_total,
            "prompt_lengths": prompt_lengths,
            "prompt_token_ids_sha256": PROMPT_DIGEST,
        }
        generated_fingerprint = {
            "schema": "wkvm.generated_output_token_ids.sha256.v1",
            "request_count": batch,
            "request_ids": [f"bench-{batch}-{index}" for index in range(batch)],
            "output_token_count": batch * output_tokens,
            "output_token_counts": [output_tokens] * batch,
            "request_output_token_ids_sha256": OUTPUT_DIGEST,
        }
        row = {
            "B": batch,
            "success_count": batch,
            "error_count": 0,
            "error": None,
            "prompt_lengths": prompt_lengths,
            "prompt_fingerprint": prompt_fingerprint,
            "output_token_counts": [output_tokens] * batch,
            "generated_output_fingerprint": generated_fingerprint,
            "batch_wall_s": batch_wall,
            "batch_wall_scope": "synchronous_batch_completion",
            "e2e_output_tok_s": batch * output_tokens / batch_wall,
            "cohort_prefill_wall_s": cohort_wall,
            "cohort_input_tok_s": prompt_total / cohort_wall,
            "gpu_memory": memory,
            "peak_engine_delta_gib": (peak_used - baseline) / 1024.0,
            "green": True,
        }
        if engine == "wkvm-native":
            row.update(
                {
                    "cohort_input_token_count": prompt_total,
                    "cohort_prefill_scope": "same_run_max_request_ttft",
                    "p50_ttft_s": 1.0 + repeat * 0.01,
                    "p95_ttft_s": 1.5 + repeat * 0.01,
                    "max_ttft_s": cohort_wall,
                    "ttft_request_count": batch,
                    "decode_interval_s": decode_interval,
                    "decode_interval_scope": "batch_earliest_first_to_latest_last",
                    "agg_decode_tok_s": batch * (output_tokens - 1) / decode_interval,
                }
            )
            provenance = {
                "schema": "wkvm.native_gemma_bench.provenance.v1",
                "benchmark": {
                    "git_commit": COMMIT,
                    "wkvm_package_version": "0.0.1",
                    "git_worktree_dirty": False,
                    "source_identity": SOURCE_IDENTITY,
                    "pre_run_source_identity_sha256": SOURCE_IDENTITY_SHA256,
                    "source_identity_unchanged_during_run": True,
                },
                "environment": {
                    "python_version": "3.11.8",
                    "cuda_visible_devices": str(GPU_INDEX),
                    "packages": {"wkvm": "0.0.1", "torch": "2.11.0"},
                },
                "gpu": {
                    "device_selector": device_selector,
                    "device_index": device_index,
                    "device_uuid": device_uuid,
                    "gpu_name": GPU_MODEL,
                    "driver_version": DRIVER,
                    "memory_total_mib": 81_920,
                },
            }
            schema = "wkvm.native_gemma_bench.v1"
            semantics = "routed_span_approximate"
        else:
            row["cohort_input_tokens"] = prompt_total
            if engine == "vllm":
                row.update(
                    {
                        "cohort_prefill_scope": "max_request_ttft_synchronous_cohort",
                        "cohort_prefill_comparable": True,
                        "p50_ttft_s": 0.8 + repeat * 0.01,
                        "p95_ttft_s": 1.2 + repeat * 0.01,
                        "max_ttft_s": cohort_wall,
                        "ttft_metric_count": batch,
                        "decode_interval_s": decode_interval,
                        "decode_interval_scope": "batch_earliest_first_to_latest_last",
                        "decode_timing_method": "same_run_request_metrics",
                        "decode_timing_comparable": True,
                        "agg_decode_tok_s": batch * (output_tokens - 1) / decode_interval,
                    }
                )
                version = "0.25.1"
            else:
                row.update(
                    {
                        "cohort_prefill_scope": "separate_run_batch_wall",
                        "cohort_prefill_comparable": False,
                        "p50_ttft_s": None,
                        "p95_ttft_s": None,
                        "max_ttft_s": None,
                        "decode_interval_s": batch_wall - cohort_wall,
                        "decode_interval_scope": "separate_run_wall_time_subtraction",
                        "decode_timing_method": "separate_run_subtraction",
                        "decode_timing_comparable": False,
                        "agg_decode_tok_s": 999.0,
                        "separate_timing_probe_order": "full_then_max_tokens_1",
                    }
                )
                version = "0.5.15.post1"
            provenance = {
                "schema": "wkvm.incumbent_gemma_bench.provenance.v1",
                "benchmark": {
                    "git_commit": COMMIT,
                    "git_worktree_dirty": False,
                    "source_identity": SOURCE_IDENTITY,
                    "pre_run_source_identity_sha256": SOURCE_IDENTITY_SHA256,
                    "source_identity_unchanged_during_run": True,
                },
                "runtime": {
                    "python_version": "3.11.8",
                    "cuda_visible_devices": str(GPU_INDEX),
                    "packages": {engine: version, "torch": "2.11.0"},
                },
                "engine": {
                    "label": engine,
                    "version": version,
                    "version_source": "imported_package",
                },
                "gpu": {
                    "index": device_index,
                    "uuid": device_uuid,
                    "name": GPU_MODEL,
                    "driver_version": DRIVER,
                    "memory_total_mib": 81_920,
                },
            }
            schema = "wkvm.incumbent_gemma_bench.v1"
            semantics = "full_kv"
        return {
            "schema": schema,
            "engine": engine,
            "semantics": semantics,
            "context_tokens_per_session": 16,
            "decode_tokens_per_session": output_tokens,
            "prompt_lengths_mode": "uniform",
            "dtype": "bfloat16",
            "device": "cuda" if engine == "wkvm-native" else None,
            "mem_cap_gib": 76.0,
            "headroom_gib": 4.0,
            "max_baseline_gpu_used_gib": 1.0,
            "model_path": "/models/gemma-4-E4B-it",
            "model_identity": MODEL_IDENTITY,
            "prompt_token_source": "synthetic",
            "uses_hf_tokenizer": False,
            "warmup": False,
            "git_commit": COMMIT,
            "launch_command": (
                f"python benchmark.py --engine {engine} --ctx 16 "
                f"--json result-{engine}-{repeat}.json"
            ),
            "provenance": provenance,
            (
                "config" if engine == "wkvm-native" else "engine_config"
            ): (
                {"m_slots": 32, "route_chunk": 8, "decode_microbatch_rows": batch}
                if engine == "wkvm-native"
                else {
                    "engine": engine,
                    "max_num_seqs": batch,
                    "prefix_caching": False,
                    **(
                        {"max_num_batched_tokens": batch * 16}
                        if engine == "vllm"
                        else {"chunked_prefill_size": 16}
                    ),
                }
            ),
            "rows": [row],
        }

    def write_payload(self, engine: str, repeat: int, *, batch: int = 2) -> Path:
        path = self.root / f"{engine}-b{batch}-r{repeat}.json"
        path.write_text(json.dumps(self.payload(engine, repeat, batch=batch)))
        return path

    def cohort(self, *, engines: tuple[str, ...] = ("wkvm-native", "vllm", "sglang"), repeats: int = 3) -> list[Path]:
        return [
            self.write_payload(engine, repeat)
            for engine in engines
            for repeat in range(repeats)
        ]

    def rewrite(self, path: Path, mutate) -> None:
        payload = json.loads(path.read_text())
        mutate(payload)
        path.write_text(json.dumps(payload))

    def test_builds_strict_summary_and_excludes_sglang_decode_ratio(self) -> None:
        paths = self.cohort()

        summary = build_summary(paths)
        markdown = render_markdown(summary)

        self.assertEqual(summary["status"], "pass")
        self.assertEqual(summary["minimum_samples_per_engine_batch"], 3)
        self.assertFalse(
            summary["ten_x_e2e_claim_gate"][
                "any_batch_passes_all_incumbents"
            ]
        )
        self.assertEqual(len(summary["groups"]), 3)
        wkvm = next(group for group in summary["groups"] if group["engine"] == "wkvm-native")
        sglang = next(group for group in summary["groups"] if group["engine"] == "sglang")
        self.assertEqual(wkvm["semantics"], "routed_span_approximate")
        self.assertEqual(wkvm["metrics"]["batch_wall_s"], {"count": 3, "median": 5.0, "min": 4.0, "max": 6.0})
        self.assertTrue(wkvm["decode"]["ratio_eligible"])
        self.assertFalse(sglang["decode"]["ratio_eligible"])
        self.assertIsNone(sglang["metrics"]["comparable_decode_tok_s"])
        self.assertEqual(
            {item["numerator_engine"] for item in summary["comparisons"]},
            {"wkvm-native"},
        )
        self.assertEqual(
            {item["denominator_engine"] for item in summary["comparisons"]},
            {"vllm", "sglang"},
        )
        sglang_ratios = [
            item
            for item in summary["comparisons"]
            if "sglang" in {item["numerator_engine"], item["denominator_engine"]}
        ]
        self.assertTrue(sglang_ratios)
        self.assertTrue(all(item["median_cohort_input_ratio"] is None for item in sglang_ratios))
        self.assertTrue(all(item["median_comparable_decode_ratio"] is None for item in sglang_ratios))
        self.assertTrue(all(item["decode_ratio_exclusion"] == "separate_run_subtraction" for item in sglang_ratios))
        self.assertIn("routed_span_approximate", markdown)
        self.assertIn("full_kv", markdown)
        self.assertIn("excluded (separate_run_subtraction)", markdown)
        self.assertIn("Every ratio is `wkvm-native / incumbent`", markdown)
        self.assertIn("## 10x E2E Claim Gate", markdown)
        self.assertIn("`**/__pycache__/**`", markdown)
        self.assertIn("`.cache/**`", markdown)
        for path in paths:
            self.assertIn(path.as_posix(), markdown)

    def test_writes_markdown_and_machine_readable_json(self) -> None:
        paths = self.cohort()
        markdown_path = self.root / "report.md"
        summary_path = self.root / "report.json"

        summary = write_report(
            paths,
            markdown_path=markdown_path,
            summary_json_path=summary_path,
        )

        self.assertEqual(json.loads(summary_path.read_text()), summary)
        self.assertEqual(markdown_path.read_text(), render_markdown(summary))
        self.assertIn("result-vllm-1.json", markdown_path.read_text())
        artifact = summary["contract"]["artifacts"][0]
        self.assertIn("launch_command", artifact)
        self.assertIn("provenance", artifact)
        self.assertEqual(
            summary["contract"]["model_identity_sha256"],
            MODEL_MANIFEST_SHA256,
        )
        self.assertEqual(summary["contract"]["gpu"]["uuid"], GPU_UUID)
        self.assertEqual(
            summary["contract"]["output_fingerprints_by_batch"],
            {"2": OUTPUT_DIGEST},
        )
        self.assertEqual(
            summary["contract"]["policy"]["gpu_memory_sample_interval_s"],
            0.1,
        )

    def test_requires_three_samples_per_engine_batch_by_default(self) -> None:
        paths = self.cohort(repeats=2)

        with self.assertRaisesRegex(ReliableReportError, "require at least 3"):
            build_summary(paths)

        with self.assertRaisesRegex(ReliableReportError, "integer >= 3"):
            build_summary(paths, min_samples=2)

    def test_rejects_shape_and_prompt_fingerprint_mismatches(self) -> None:
        for field, value, message in (
            ("context_tokens_per_session", 32, "same-shape requirement failed"),
            ("prompt_digest", "d" * 64, "same-prompt-fingerprint requirement failed"),
        ):
            with self.subTest(field=field):
                paths = self.cohort()
                if field == "prompt_digest":
                    self.rewrite(
                        paths[-1],
                        lambda data: data["rows"][0]["prompt_fingerprint"].update(
                            {"prompt_token_ids_sha256": value}
                        ),
                    )
                else:
                    self.rewrite(paths[-1], lambda data: data.update({field: value}))
                with self.assertRaisesRegex(ReliableReportError, message):
                    build_summary(paths)
                for path in paths:
                    path.unlink(missing_ok=True)

    def test_rejects_output_fingerprint_mismatch(self) -> None:
        paths = self.cohort()
        self.rewrite(
            paths[-1],
            lambda data: data["rows"][0]["generated_output_fingerprint"].update(
                {"request_output_token_ids_sha256": "e" * 64}
            ),
        )

        with self.assertRaisesRegex(
            ReliableReportError,
            "same-output-fingerprint requirement failed",
        ):
            build_summary(paths)

    def test_rejects_unsuccessful_or_inexact_token_accounting(self) -> None:
        mutations = (
            (
                lambda data: data["rows"][0].update({"success_count": 1}),
                "not fully successful",
            ),
            (
                lambda data: data["rows"][0]["generated_output_fingerprint"].update(
                    {"output_token_counts": [4, 3]}
                ),
                "exactly 4 output tokens",
            ),
            (
                lambda data: data["rows"][0].update({"e2e_output_tok_s": 999.0}),
                "disagrees with output token accounting",
            ),
        )
        for mutation, message in mutations:
            with self.subTest(message=message):
                paths = self.cohort()
                self.rewrite(paths[-1], mutation)
                with self.assertRaisesRegex(ReliableReportError, message):
                    build_summary(paths)
                for path in paths:
                    path.unlink(missing_ok=True)

    def test_rejects_gpu_model_or_driver_mismatch(self) -> None:
        for field, value in (("name", "NVIDIA H100"), ("driver_version", "999.0")):
            with self.subTest(field=field):
                paths = self.cohort()
                memory_field = "gpu_name" if field == "name" else field
                self.rewrite(
                    paths[-1],
                    lambda data: (
                        data["provenance"]["gpu"].update({field: value}),
                        data["rows"][0]["gpu_memory"].update(
                            {memory_field: value}
                        ),
                    ),
                )
                with self.assertRaisesRegex(
                    ReliableReportError,
                    "physical GPU UUID/model/driver comparability failed",
                ):
                    build_summary(paths)
                for path in paths:
                    path.unlink(missing_ok=True)

    def test_rejects_invalid_memory_telemetry(self) -> None:
        mutations = (
            lambda memory: memory.update({"query_error_count": 1, "error": "query failed"}),
            lambda memory: memory.update({"sample_count": 1}),
            lambda memory: memory.update({"peak_delta_mib": 123}),
        )
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                paths = self.cohort()
                self.rewrite(paths[-1], lambda data: mutation(data["rows"][0]["gpu_memory"]))
                with self.assertRaisesRegex(ReliableReportError, "gpu_memory"):
                    build_summary(paths)
                for path in paths:
                    path.unlink(missing_ok=True)

    def test_rejects_rows_that_fail_configured_memory_gate(self) -> None:
        def exceed_gate(data: dict) -> None:
            row = data["rows"][0]
            memory = row["gpu_memory"]
            delta_mib = 74 * 1024
            memory["peak_delta_mib"] = delta_mib
            memory["peak_used_mib"] = memory["baseline_used_mib"] + delta_mib
            row["peak_engine_delta_gib"] = 74.0

        for mutation in (
            lambda data: data["rows"][0].update({"green": False}),
            exceed_gate,
        ):
            with self.subTest(mutation=mutation):
                paths = self.cohort()
                self.rewrite(paths[-1], mutation)
                with self.assertRaisesRegex(ReliableReportError, "failed memory gate"):
                    build_summary(paths)
                for path in paths:
                    path.unlink(missing_ok=True)

    def test_rejects_missing_launch_or_provenance(self) -> None:
        for field, message in (
            ("launch_command", "missing non-empty launch_command"),
            ("provenance", "missing structured provenance"),
        ):
            with self.subTest(field=field):
                paths = self.cohort()
                self.rewrite(paths[-1], lambda data: data.pop(field))
                with self.assertRaisesRegex(ReliableReportError, message):
                    build_summary(paths)
                for path in paths:
                    path.unlink(missing_ok=True)

    def test_rejects_noncomparable_vllm_decode_but_allows_sglang(self) -> None:
        paths = self.cohort()
        vllm_path = next(path for path in paths if path.name.startswith("vllm"))
        self.rewrite(
            vllm_path,
            lambda data: data["rows"][0].update(
                {
                    "decode_timing_method": "separate_run_subtraction",
                    "decode_timing_comparable": False,
                }
            ),
        )

        with self.assertRaisesRegex(ReliableReportError, "same-run decode interval"):
            build_summary(paths)

    def test_sglang_prefill_and_decode_are_always_excluded(self) -> None:
        paths = self.cohort()
        sglang_path = next(path for path in paths if path.name.startswith("sglang"))
        self.rewrite(
            sglang_path,
            lambda data: data["rows"][0].update(
                {
                    "p50_ttft_s": 0.5,
                    "p95_ttft_s": 0.7,
                    "max_ttft_s": data["rows"][0]["cohort_prefill_wall_s"],
                    "ttft_metric_count": data["rows"][0]["B"],
                    "decode_interval_scope": "batch_earliest_first_to_latest_last",
                    "decode_timing_method": "same_run_request_metrics",
                    "decode_timing_comparable": True,
                }
            ),
        )

        with self.assertRaisesRegex(
            ReliableReportError,
            "SGLang separate-run prefill cannot report request TTFT",
        ):
            build_summary(paths)

    def test_rejects_policy_config_or_launch_drift_across_repeats(self) -> None:
        cases = (
            (
                lambda data: data.update({"mem_cap_gib": 75.0}),
                "common benchmark policy differs",
            ),
            (
                lambda data: data.update({"warmup": True}),
                "reliable cold comparison requires warmup=false",
            ),
            (
                lambda data: data["engine_config"].update({"max_num_seqs": 99}),
                "engine configuration differs across repeats",
            ),
            (
                lambda data: data.update(
                    {
                        "launch_command": data["launch_command"].replace(
                            "--ctx 16", "--ctx 32"
                        )
                    }
                ),
                "launch policy/config differs across repeats",
            ),
        )
        for mutation, message in cases:
            with self.subTest(message=message):
                paths = self.cohort()
                self.rewrite(paths[-1], mutation)
                with self.assertRaisesRegex(ReliableReportError, message):
                    build_summary(paths)
                for path in paths:
                    path.unlink(missing_ok=True)

    def test_requires_explicit_incumbent_prefill_scheduler_controls(self) -> None:
        for engine, field in (
            ("vllm", "max_num_batched_tokens"),
            ("sglang", "chunked_prefill_size"),
        ):
            with self.subTest(engine=engine):
                paths = self.cohort()
                path = next(item for item in paths if item.name.startswith(engine))
                self.rewrite(
                    path,
                    lambda data: data["engine_config"].pop(field),
                )
                with self.assertRaisesRegex(
                    ReliableReportError,
                    f"engine_config.{field} must be an explicit positive integer",
                ):
                    build_summary(paths)
                for item in paths:
                    item.unlink(missing_ok=True)

    def test_requires_all_three_public_comparison_engines(self) -> None:
        paths = self.cohort(engines=("wkvm-native", "vllm"))

        with self.assertRaisesRegex(
            ReliableReportError,
            "requires wkvm-native, vllm, and sglang evidence",
        ):
            build_summary(paths)

    def test_rejects_physical_gpu_or_compute_selector_drift(self) -> None:
        cases = (
            (
                lambda data: (
                    data["provenance"]["gpu"].update({"uuid": "GPU-other"}),
                    data["rows"][0]["gpu_memory"].update(
                        {"device_uuid": "GPU-other"}
                    ),
                ),
                "physical GPU UUID/model/driver comparability failed",
            ),
            (
                lambda data: data["provenance"]["runtime"].update(
                    {"cuda_visible_devices": "7"}
                ),
                "monitored GPU does not match CUDA_VISIBLE_DEVICES index",
            ),
        )
        for mutation, message in cases:
            with self.subTest(message=message):
                paths = self.cohort()
                self.rewrite(paths[-1], mutation)
                with self.assertRaisesRegex(ReliableReportError, message):
                    build_summary(paths)
                for path in paths:
                    path.unlink(missing_ok=True)

    def test_rejects_non_cuda_native_compute_device(self) -> None:
        paths = self.cohort()
        native_path = next(
            path for path in paths if path.name.startswith("wkvm-native")
        )
        self.rewrite(native_path, lambda data: data.update({"device": "cpu"}))

        with self.assertRaisesRegex(
            ReliableReportError,
            "native compute device must be cuda",
        ):
            build_summary(paths)

    def test_rejects_source_or_model_identity_drift(self) -> None:
        def mutate_source(data: dict) -> None:
            benchmark = data["provenance"]["benchmark"]
            identity = benchmark["source_identity"]
            identity["worktree_manifest_sha256"] = "5" * 64
            fields = {
                key: identity[key]
                for key in (
                    "git_commit",
                    "git_head_tree",
                    "git_status_sha256",
                    "git_tracked_diff_sha256",
                    "worktree_manifest_sha256",
                )
            }
            identity["identity_sha256"] = canonical_sha256(fields)
            benchmark["pre_run_source_identity_sha256"] = identity[
                "identity_sha256"
            ]

        def mutate_model(data: dict) -> None:
            identity = data["model_identity"]
            identity["files"][0]["sha256"] = "6" * 64
            identity["manifest_sha256"] = canonical_sha256(identity["files"])

        for mutation, message in (
            (mutate_source, "exact source/worktree identities differ"),
            (mutate_model, "model checkpoint manifests differ"),
        ):
            with self.subTest(message=message):
                paths = self.cohort()
                self.rewrite(paths[-1], mutation)
                with self.assertRaisesRegex(ReliableReportError, message):
                    build_summary(paths)
                for path in paths:
                    path.unlink(missing_ok=True)

    def test_rejects_changed_source_and_busy_gpu_baseline(self) -> None:
        def make_source_dirty(data: dict) -> None:
            benchmark = data["provenance"]["benchmark"]
            identity = benchmark["source_identity"]
            identity["git_status_sha256"] = "5" * 64
            identity["git_worktree_dirty"] = True
            benchmark["git_worktree_dirty"] = True
            fields = {
                key: identity[key]
                for key in (
                    "git_commit",
                    "git_head_tree",
                    "git_status_sha256",
                    "git_tracked_diff_sha256",
                    "worktree_manifest_sha256",
                )
            }
            identity["identity_sha256"] = canonical_sha256(fields)
            benchmark["pre_run_source_identity_sha256"] = identity[
                "identity_sha256"
            ]

        cases = (
            (
                lambda data: data["provenance"]["benchmark"].update(
                    {"source_identity_unchanged_during_run": False}
                ),
                "source identity must be unchanged",
            ),
            (make_source_dirty, "source worktree must be clean"),
            (
                lambda data: data["rows"][0]["gpu_memory"].update(
                    {
                        "baseline_used_mib": 75_000,
                        "peak_used_mib": 76_000,
                        "peak_delta_mib": 1_000,
                    }
                ),
                "exceeds idle ceiling",
            ),
        )
        for mutation, message in cases:
            with self.subTest(message=message):
                paths = self.cohort()
                self.rewrite(paths[-1], mutation)
                with self.assertRaisesRegex(ReliableReportError, message):
                    build_summary(paths)
                for path in paths:
                    path.unlink(missing_ok=True)

    def test_rejects_non_idle_policy_and_hidden_source_exclusions(self) -> None:
        cases = (
            (
                lambda data: data.update({"max_baseline_gpu_used_gib": 1.01}),
                "must be <= 1.0 GiB",
            ),
            (
                lambda data: data["provenance"]["benchmark"]["source_identity"].update(
                    {"excluded_paths": ["wkvm/gemma_engine.py"]}
                ),
                "unsupported extra exclusions",
            ),
        )
        for mutation, message in cases:
            with self.subTest(message=message):
                paths = self.cohort()
                for path in paths:
                    self.rewrite(path, mutation)
                with self.assertRaisesRegex(ReliableReportError, message):
                    build_summary(paths)
                for path in paths:
                    path.unlink(missing_ok=True)

    def test_rejects_different_gpu_memory_sample_intervals(self) -> None:
        paths = self.cohort()
        self.rewrite(
            paths[-1],
            lambda data: data["rows"][0]["gpu_memory"].update(
                {"sample_interval_s": 0.2}
            ),
        )

        with self.assertRaisesRegex(
            ReliableReportError,
            "GPU memory sample intervals differ",
        ):
            build_summary(paths)

    def test_rejects_sglang_probe_before_cold_full_run(self) -> None:
        paths = self.cohort()
        sglang_path = next(
            path for path in paths if path.name.startswith("sglang")
        )
        self.rewrite(
            sglang_path,
            lambda data: data["rows"][0].update(
                {"separate_timing_probe_order": "max_tokens_1_then_full"}
            ),
        )

        with self.assertRaisesRegex(
            ReliableReportError,
            "cold full run must precede",
        ):
            build_summary(paths)

    def test_wkvm_incumbent_ratio_orientation_is_numerator_over_denominator(self) -> None:
        paths = self.cohort()
        for path in paths:
            if not path.name.startswith("vllm"):
                continue

            def slow_vllm(data: dict) -> None:
                row = data["rows"][0]
                row["batch_wall_s"] *= 2
                row["e2e_output_tok_s"] = (
                    sum(row["output_token_counts"]) / row["batch_wall_s"]
                )

            self.rewrite(path, slow_vllm)

        summary = build_summary(paths)
        comparison = next(
            item
            for item in summary["comparisons"]
            if item["denominator_engine"] == "vllm"
        )
        self.assertEqual(comparison["numerator_engine"], "wkvm-native")
        self.assertEqual(comparison["ratio_definition"], "numerator_over_denominator")
        self.assertAlmostEqual(comparison["median_e2e_output_ratio"], 2.0)
        self.assertAlmostEqual(
            comparison["conservative_e2e_output_ratio"],
            4.0 / 3.0,
        )

    def test_ten_x_gate_requires_worst_repeat_win_against_both_incumbents(self) -> None:
        paths = self.cohort()
        for path in paths:
            if path.name.startswith("wkvm-native"):
                continue

            def slow_incumbent(data: dict) -> None:
                row = data["rows"][0]
                row["batch_wall_s"] *= 20
                row["e2e_output_tok_s"] = (
                    sum(row["output_token_counts"]) / row["batch_wall_s"]
                )

            self.rewrite(path, slow_incumbent)

        summary = build_summary(paths)
        gate = summary["ten_x_e2e_claim_gate"]

        self.assertTrue(gate["any_batch_passes_all_incumbents"])
        self.assertTrue(gate["batches"][0]["passes_all_incumbents"])
        self.assertTrue(
            all(
                comparison["ten_x_e2e_claim_pass"]
                for comparison in summary["comparisons"]
            )
        )
        self.assertIn("**PASS**", render_markdown(summary))

    def test_parser_requires_both_output_artifacts(self) -> None:
        parser = build_arg_parser()
        help_text = parser.format_help()

        self.assertIn("--markdown", help_text)
        self.assertIn("--summary-json", help_text)
        self.assertIn("PATH=<venv>/bin:$PATH", help_text)
        self.assertIn("--no-warmup", help_text)
        with self.assertRaises(SystemExit):
            parser.parse_args(["input.json"])

    def test_model_identity_hashes_sorted_file_manifest(self) -> None:
        model_root = self.root / "model"
        model_root.mkdir()
        (model_root / "model.safetensors").write_bytes(b"weights")
        (model_root / "config.json").write_text("{}")
        cache_file = model_root / ".cache" / "huggingface" / "download.lock"
        cache_file.parent.mkdir(parents=True)
        cache_file.write_text("volatile")

        identity = model_checkpoint_identity(model_root)
        cache_file.write_text("changed")
        after_cache_change = model_checkpoint_identity(model_root)

        self.assertIsNone(identity["error"])
        self.assertEqual(
            [entry["path"] for entry in identity["files"]],
            ["config.json", "model.safetensors"],
        )
        self.assertEqual(
            identity["manifest_sha256"],
            canonical_sha256(identity["files"]),
        )
        self.assertEqual(identity["excluded_path_patterns"], [".cache/**"])
        self.assertEqual(
            identity["manifest_sha256"],
            after_cache_change["manifest_sha256"],
        )

    def test_source_identity_excludes_only_generated_results(self) -> None:
        repo = self.root / "source-repo"
        result = repo / "experiments" / "results" / "run.json"
        source = repo / "wkvm" / "engine.py"
        result.parent.mkdir(parents=True)
        source.parent.mkdir(parents=True)
        result.write_text("old result")
        source.write_text("version = 1\n")
        (repo / ".gitignore").write_text("hidden_source.py\n")
        subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
        subprocess.run(
            ["git", "config", "user.email", "benchmark@example.invalid"],
            cwd=repo,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Benchmark Test"],
            cwd=repo,
            check=True,
        )
        subprocess.run(["git", "add", "."], cwd=repo, check=True)
        subprocess.run(
            ["git", "commit", "-qm", "fixture"],
            cwd=repo,
            check=True,
        )

        before = source_worktree_identity(repo)
        result.write_text("new generated result")
        after_result = source_worktree_identity(repo)
        (repo / "hidden_source.py").write_text("ignored = True\n")
        after_ignored_source = source_worktree_identity(repo)
        source.write_text("version = 2\n")
        after_source = source_worktree_identity(repo)

        self.assertIsNone(before["error"])
        self.assertEqual(before["identity_sha256"], after_result["identity_sha256"])
        self.assertNotEqual(
            after_result["identity_sha256"],
            after_ignored_source["identity_sha256"],
        )
        self.assertTrue(after_ignored_source["git_worktree_dirty"])
        self.assertNotEqual(
            after_ignored_source["identity_sha256"],
            after_source["identity_sha256"],
        )
        self.assertEqual(
            before["excluded_path_patterns"],
            SOURCE_EXCLUDED_PATH_PATTERNS,
        )
        self.assertEqual(before["excluded_paths"], [])


if __name__ == "__main__":
    unittest.main()
