from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.phase3_gemma_report import (
    PROFILE_SPECS,
    Phase3ReportError,
    build_summary,
    render_markdown,
    write_report,
)


SOURCE_SHA = "1" * 64
MODEL_SHA = "2" * 64
PROMPT_HASHES = {"prefill": "3" * 64, "schedule": "4" * 64}
OUTPUT_HASHES = {"prefill": "5" * 64, "schedule": "6" * 64}


class TestPhase3GemmaReport(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def payload(self, profile: str, repeat: int) -> dict:
        spec = PROFILE_SPECS[profile]
        prefill_rates = {
            "prefill-baseline": 1000.0,
            "prefill-packed": 1040.0,
            "prefill-routed-packets": 1060.0,
            "prefill-native-gqa": 1055.0,
            "prefill-combined": 1065.0,
        }
        input_rate = prefill_rates.get(profile, 900.0) + repeat
        schedule_lane = profile == "schedule-lane8"
        routed_packets = spec.batched_routed_packets
        prompt_total = spec.batch * spec.context_tokens
        output_total = spec.batch * spec.output_tokens
        batch_wall = 3.0 if not schedule_lane else 2.9
        row = {
            "B": spec.batch,
            "success_count": spec.batch,
            "error_count": 0,
            "error": None,
            "green": True,
            "torch_reserved_green": True,
            "uses_hf_transformer_forward": False,
            "uses_hf_model_construction": False,
            "native_gemma_checkpoint_loader": True,
            "cohort_input_tok_s": input_rate,
            "cohort_prefill_wall_s": prompt_total / input_rate,
            "prefill_time_p50_s": 2.0,
            "prefill_time_p95_s": 2.1,
            "p50_ttft_s": 2.2 if not schedule_lane else 2.0,
            "p95_ttft_s": 2.3 if not schedule_lane else 2.5,
            "max_ttft_s": prompt_total / input_rate,
            "batch_wall_s": batch_wall,
            "e2e_output_tok_s": output_total / batch_wall,
            "peak_reserved_gib": 14.0,
            "peak_engine_delta_gib": 15.0,
            "prompt_total_tokens": prompt_total,
            "cohort_input_token_count": prompt_total,
            "prompt_token_source": "synthetic",
            "prompt_lengths": [spec.context_tokens] * spec.batch,
            "prompt_token_ids_sha256": PROMPT_HASHES[spec.family],
            "prompt_fingerprint": {
                "schema": "wkvm.prompt_token_ids.sha256.v1",
                "prompt_token_source": "synthetic",
                "prompt_count": spec.batch,
                "prompt_lengths": [spec.context_tokens] * spec.batch,
                "prompt_total_tokens": prompt_total,
                "prompt_token_ids_sha256": PROMPT_HASHES[spec.family],
            },
            "request_output_token_ids_sha256": OUTPUT_HASHES[spec.family],
            "generated_output_fingerprint": {
                "schema": "wkvm.generated_output_token_ids.sha256.v1",
                "request_count": spec.batch,
                "request_ids": sorted(
                    [f"bench-{spec.batch}-{index}" for index in range(spec.batch)],
                    key=lambda value: value.encode("utf-8"),
                ),
                "output_token_count": output_total,
                "output_token_counts": [spec.output_tokens] * spec.batch,
                "request_output_token_ids_sha256": OUTPUT_HASHES[spec.family],
            },
            "generated_output_request_count": spec.batch,
            "generated_output_token_count": output_total,
            "generated_output_token_counts": [spec.output_tokens] * spec.batch,
            "generated_output_request_ids": sorted(
                [f"bench-{spec.batch}-{index}" for index in range(spec.batch)],
                key=lambda value: value.encode("utf-8"),
            ),
            "generated_output_fingerprint_schema": (
                "wkvm.generated_output_token_ids.sha256.v1"
            ),
            "routed_packets": {
                "enabled": routed_packets,
                "workspace_max_bytes": 67_108_864,
                **(
                    {
                        "packet_batches": 4,
                        "d2h_copies": 4,
                        "packet_folds": 12,
                        "packet_request_rows": 8,
                        "workspace_pinned_host_buffer_bytes": 4096,
                        "capacity_fallback_batches": 0,
                    }
                    if routed_packets
                    else {}
                ),
            },
            "routed_packet_evidence_passed": True,
            "completion_prefill_lane_size": spec.completion_prefill_lane_size,
            "completion_prefill_lane_starts": 2 if schedule_lane else 0,
            "completion_prefill_lane_completions": 2 if schedule_lane else 0,
            "completion_prefill_lane_cancellations": 0,
            "native_forward_timing": {
                "available": True,
                "dense_gqa_prefill_calls": (
                    24 if spec.attention_backend == "triton_dense_gqa" else 0
                ),
                "dense_gqa_prefill_fallbacks": 0,
                "dense_gqa_decode_calls": 0,
                "dense_gqa_decode_fallbacks": 0,
            },
        }
        return {
            "schema": "wkvm.native_gemma_bench.v1",
            "engine": "wkvm-native",
            "context_tokens_per_session": spec.context_tokens,
            "decode_tokens_per_session": spec.output_tokens,
            "concurrency": [spec.batch],
            "prompt_lengths_mode": "uniform",
            "prompt_token_source": "synthetic",
            "dtype": "bfloat16",
            "device": "cuda",
            "attn": "sdpa",
            "warmup": False,
            "fatal_error": None,
            "mem_cap_gib": 24.0,
            "headroom_gib": 4.0,
            "max_baseline_gpu_used_gib": 1.0,
            "native_gemma_checkpoint_loader": True,
            "uses_hf_transformer_forward": False,
            "uses_hf_model_construction": False,
            "native_gemma_attention_backend": spec.attention_backend,
            "native_gemma_projection_backend": spec.projection_backend,
            "native_gemma_weight_backend": "hf_live",
            "native_gemma_release_hf_decoder_layers": False,
            "native_gemma_config_loader": True,
            "uses_hf_config": False,
            "uses_hf_tokenizer": False,
            "use_native_gemma_forward": True,
            "token_pool_attention_enabled": True,
            "cuda_phase_metrics_enabled": False,
            "token_pool_triton_env": {
                "WKVM_ENABLE_TOKEN_POOL_PAGED_SPLIT_TRITON": "1",
                "WKVM_ENABLE_TOKEN_POOL_PAGED_TRITON": "1",
                "WKVM_ENABLE_TOKEN_POOL_TRITON": "1",
                "WKVM_TOKEN_POOL_SLIDING_PAGED_METADATA_ONLY": "1",
                "WKVM_TOKEN_POOL_TRITON_STRICT": "1",
            },
            "batched_routed_packets": routed_packets,
            "completion_prefill_lane_size": spec.completion_prefill_lane_size,
            "native_no_hf_requirement": {"passed": True, "violations": []},
            "routed_packet_evidence": {
                "required": routed_packets,
                "passed": True,
            },
            "generated_output_fingerprint_coverage": {
                "complete": True,
                "successful_rows": 1,
                "fingerprinted_successful_rows": 1,
            },
            "git_commit": "a" * 40,
            "config": {
                "slots": spec.batch,
                "route_chunk": 512,
                "chunk": 2048,
                "prefill_microbatch_rows": 8,
                "decode_microbatch_rows": spec.batch,
                "decode_microbatch_bytes": None,
                "decode_batch_planner": "scheduler",
                "decode_workspace_bytes": None,
                "decode_workspace_width_bucket": 16,
                "cuda_phase_metrics": False,
                "persistent_exact_decode": True,
                "persistent_padded_decode": True,
                "persistent_padded_decode_steps": spec.output_tokens,
                "persistent_padded_full_attention_rows": None,
                "persistent_padded_sliding_metadata_padding": True,
                "persistent_padded_decode_cuda_graph": True,
                "persistent_padded_decode_graph_warmup_iters": 0,
                "native_gemma_projection_backend": spec.projection_backend,
                "native_gemma_attention_backend": spec.attention_backend,
                "native_gemma_weight_backend": "hf_live",
                "native_gemma_release_hf_decoder_layers": False,
                "use_native_gemma_forward": True,
                "uses_hf_config": False,
                "native_gemma_config_loader": True,
                "batched_routed_packets": routed_packets,
                "routed_packet_workspace_bytes": 67_108_864,
                "completion_prefill_lane_size": spec.completion_prefill_lane_size,
                "synthetic_prompts": True,
                "synthetic_vocab_size": 262_144,
                "enable_token_pool_attention": True,
                "enable_token_pool_metadata": None,
                "enable_token_pool_triton": True,
                "enable_token_pool_paged_triton": True,
                "enable_token_pool_paged_split_triton": True,
                "token_pool_capacity": 65_536,
                "token_pool_max_context_len": 16_640,
                "token_pool_paged_block_size": 16,
                "token_pool_triton_strict": True,
                "token_pool_sliding_paged_metadata_only": True,
                "token_pool_triton_env": {
                    "WKVM_ENABLE_TOKEN_POOL_PAGED_SPLIT_TRITON": "1",
                    "WKVM_ENABLE_TOKEN_POOL_PAGED_TRITON": "1",
                    "WKVM_ENABLE_TOKEN_POOL_TRITON": "1",
                    "WKVM_TOKEN_POOL_SLIDING_PAGED_METADATA_ONLY": "1",
                    "WKVM_TOKEN_POOL_TRITON_STRICT": "1",
                },
                "sink": 16,
                "window": 1024,
                "m_slots": 32,
                "token_budget": None,
            },
            "provenance": {
                "schema": "wkvm.native_gemma_bench.provenance.v1",
                "benchmark": {
                    "git_commit": "a" * 40,
                    "git_worktree_dirty": False,
                    "source_identity_unchanged_during_run": True,
                    "pre_run_source_identity_sha256": SOURCE_SHA,
                    "source_identity": {
                        "schema": "wkvm.git_worktree_identity.sha256.v1",
                        "error": None,
                        "excluded_paths": [],
                        "git_commit": "a" * 40,
                        "git_worktree_dirty": False,
                        "identity_sha256": SOURCE_SHA,
                    },
                },
                "gpu": {
                    "device_uuid": "GPU-test",
                    "gpu_name": "NVIDIA GeForce RTX 4090",
                    "driver_version": "595.71.05",
                    "memory_total_mib": 24_564,
                },
                "environment": {
                    "python_version": "3.12.13",
                    "python_implementation": "CPython",
                    "packages": {
                        "torch": "2.11.0",
                        "transformers": "5.9.0",
                        "wkvm": "0.0.1",
                    },
                },
            },
            "model_identity": {
                "schema": "wkvm.model_checkpoint_identity.sha256.v1",
                "manifest_sha256": MODEL_SHA,
                "error": None,
            },
            "gpu_memory": {
                "schema": "wkvm.whole_gpu_memory.v1",
                "scope": "whole_device",
                "source": "nvidia-smi",
                "baseline_used_mib": 512,
                "peak_used_mib": 15_872,
                "peak_delta_mib": 15_360,
                "sample_count": 10,
                "device_uuid": "GPU-test",
                "gpu_name": "NVIDIA GeForce RTX 4090",
                "driver_version": "595.71.05",
                "memory_total_mib": 24_564,
                "query_error_count": 0,
                "error": None,
            },
            "rows": [row],
        }

    def cohort(self, repeats: int = 3) -> list[Path]:
        paths = []
        for profile in PROFILE_SPECS:
            for repeat in range(1, repeats + 1):
                path = self.root / f"{profile}-r{repeat}.json"
                path.write_text(json.dumps(self.payload(profile, repeat)))
                paths.append(path)
        return paths

    def rewrite(self, path: Path, mutate) -> None:
        data = json.loads(path.read_text())
        mutate(data)
        path.write_text(json.dumps(data))

    def test_builds_strict_summary_and_selects_isolated_candidates(self) -> None:
        summary = build_summary(self.cohort())
        markdown = render_markdown(summary)

        self.assertEqual(summary["status"], "pass")
        self.assertEqual(len(summary["groups"]), len(PROFILE_SPECS))
        self.assertEqual(
            summary["selected_candidates"],
            ["prefill-routed-packets", "prefill-native-gqa"],
        )
        self.assertGreater(
            summary["schedule_comparison"]["median_p95_ttft_ratio"], 1.0
        )
        self.assertIn("Evidence gate: **PASS**", markdown)
        self.assertIn("Scheduling Tradeoff", markdown)

    def test_rejects_dirty_source_identity(self) -> None:
        paths = self.cohort()
        self.rewrite(
            paths[0],
            lambda data: data["provenance"]["benchmark"].update(
                {"git_worktree_dirty": True}
            ),
        )

        with self.assertRaisesRegex(Phase3ReportError, "source identity is dirty"):
            build_summary(paths)

    def test_rejects_cross_profile_output_mismatch(self) -> None:
        paths = self.cohort()
        mismatch = next(path for path in paths if path.name == "prefill-packed-r1.json")
        self.rewrite(
            mismatch,
            lambda data: (
                data["rows"][0].update(
                    {"request_output_token_ids_sha256": "f" * 64}
                ),
                data["rows"][0]["generated_output_fingerprint"].update(
                    {"request_output_token_ids_sha256": "f" * 64}
                ),
            ),
        )

        with self.assertRaisesRegex(Phase3ReportError, "prefill output fingerprint"):
            build_summary(paths)

    def test_rejects_inactive_packet_or_incomplete_lane_telemetry(self) -> None:
        for filename, mutate, message in (
            (
                "prefill-routed-packets-r1.json",
                lambda data: data["rows"][0]["routed_packets"].update(
                    {"packet_batches": 0}
                ),
                "zero packet batches",
            ),
            (
                "schedule-lane8-r1.json",
                lambda data: data["rows"][0].update(
                    {"completion_prefill_lane_completions": 1}
                ),
                "did not all complete",
            ),
        ):
            with self.subTest(filename=filename):
                paths = self.cohort()
                path = next(path for path in paths if path.name == filename)
                self.rewrite(path, mutate)
                with self.assertRaisesRegex(Phase3ReportError, message):
                    build_summary(paths)

    def test_rejects_missing_profile_or_insufficient_repeats(self) -> None:
        paths = self.cohort()
        without_combined = [
            path for path in paths if not path.name.startswith("prefill-combined-")
        ]
        with self.assertRaisesRegex(Phase3ReportError, "missing required"):
            build_summary(without_combined)

        with self.assertRaisesRegex(Phase3ReportError, "require 3"):
            build_summary(paths[:-1])

    def test_writes_markdown_and_machine_readable_summary(self) -> None:
        markdown_path = self.root / "report.md"
        summary_path = self.root / "summary.json"

        summary = write_report(
            self.cohort(),
            markdown_path=markdown_path,
            summary_json_path=summary_path,
        )

        self.assertEqual(json.loads(summary_path.read_text()), summary)
        self.assertEqual(markdown_path.read_text(), render_markdown(summary))


if __name__ == "__main__":
    unittest.main()
