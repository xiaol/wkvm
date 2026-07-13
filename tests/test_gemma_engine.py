import os
import sys
import unittest
from types import SimpleNamespace

from wkvm.core.config import SchedulerConfig
from wkvm.core.request import Request, RequestStatus
from wkvm.gemma_engine import GemmaEngineMetrics, _sample_argmax_token_ids


class TestGemmaEngineMetrics(unittest.TestCase):
    def test_metrics_export_is_plain_dict(self) -> None:
        metrics = GemmaEngineMetrics()
        metrics.steps = 2
        metrics.decode_timing_total_s = 0.125
        metrics.backpressure_reasons["no_free_slots"] = 1
        metrics.token_pool_decode_covered_layer_type_batches["full_attention"] = 2
        metrics.token_pool_decode_covered_layer_type_rows["full_attention"] = 8
        metrics.token_pool_decode_graph_candidate_batches = 3
        metrics.token_pool_decode_graph_static_shape_starts = 1
        metrics.token_pool_decode_graph_static_shape_reuses = 2
        metrics.token_pool_decode_graph_shape_mismatches = 1
        metrics.token_pool_decode_graph_shape_mismatch_reasons[
            "metadata_by_layer_type.full_attention.kv_indices"
        ] = 1
        metrics.token_pool_full_attention_row_rebuilds = 4
        metrics.token_pool_full_attention_row_reuses = 5
        metrics.token_pool_full_attention_row_appends = 6
        metrics.token_pool_full_attention_row_invalidations = 7
        metrics.persistent_padded_decode_cuda_graph_skips = 1
        metrics.persistent_padded_decode_cuda_graph_skip_reasons[
            "capture_failed:RuntimeError"
        ] = 1
        metrics.max_cuda_reserved_bytes = 456
        metrics.max_cuda_reserved_phase = "prefill_forward"
        metrics.cuda_current_reserved_by_phase["prefill_forward"] = 400
        metrics.cuda_peak_reserved_advances_by_phase["prefill_forward"] = 456
        data = metrics.as_dict()
        self.assertEqual(data["steps"], 2)
        self.assertEqual(data["decode_timing_total_s"], 0.125)
        self.assertEqual(data["backpressure_reasons"], {"no_free_slots": 1})
        self.assertEqual(
            data["token_pool_decode_covered_layer_type_batches"],
            {"full_attention": 2},
        )
        self.assertEqual(
            data["token_pool_decode_covered_layer_type_rows"],
            {"full_attention": 8},
        )
        self.assertEqual(data["token_pool_decode_graph_candidate_batches"], 3)
        self.assertEqual(data["token_pool_decode_graph_static_shape_starts"], 1)
        self.assertEqual(data["token_pool_decode_graph_static_shape_reuses"], 2)
        self.assertEqual(data["token_pool_decode_graph_shape_mismatches"], 1)
        self.assertEqual(
            data["token_pool_decode_graph_shape_mismatch_reasons"],
            {"metadata_by_layer_type.full_attention.kv_indices": 1},
        )
        self.assertEqual(data["token_pool_full_attention_row_rebuilds"], 4)
        self.assertEqual(data["token_pool_full_attention_row_reuses"], 5)
        self.assertEqual(data["token_pool_full_attention_row_appends"], 6)
        self.assertEqual(data["token_pool_full_attention_row_invalidations"], 7)
        self.assertEqual(data["persistent_padded_decode_cuda_graph_skips"], 1)
        self.assertEqual(
            data["persistent_padded_decode_cuda_graph_skip_reasons"],
            {"capture_failed:RuntimeError": 1},
        )
        self.assertEqual(data["max_cuda_reserved_bytes"], 456)
        self.assertEqual(data["max_cuda_reserved_phase"], "prefill_forward")
        self.assertEqual(
            data["cuda_current_reserved_by_phase"],
            {"prefill_forward": 400},
        )
        self.assertEqual(
            data["cuda_peak_reserved_advances_by_phase"],
            {"prefill_forward": 456},
        )


class TestGemmaSchedulerAssumptions(unittest.TestCase):
    def native_bench_payload_args(self, **overrides):
        values = {
            "ctx": 512,
            "prompt_lengths": "uniform",
            "out": 8,
            "synthetic_prompts": False,
            "synthetic_vocab_size": 262_144,
            "mem_cap_gib": 19.0,
            "headroom_gib": 1.0,
            "device": "cuda",
            "attn": "sdpa",
            "require_native_no_hf": True,
            "native_gemma_checkpoint_loader": True,
            "native_gemma_attention_backend": "sdpa_single_gqa",
            "native_gemma_projection_backend": "separate",
            "native_gemma_weight_backend": "hf_live",
            "native_gemma_release_hf_decoder_layers": False,
            "enable_token_pool_attention": True,
            "cuda_phase_metrics": False,
            "concurrency": [2],
            "sink": 16,
            "window": 1024,
            "m_slots": 64,
            "route_chunk": 512,
            "chunk": 2048,
            "decode_microbatch_rows": 16,
            "decode_microbatch_bytes": None,
            "decode_batch_planner": "scheduler",
            "decode_workspace_bytes": None,
            "decode_workspace_width_bucket": 16,
            "disable_persistent_exact_decode": False,
            "disable_persistent_padded_decode": False,
            "persistent_padded_decode_steps": 8,
            "persistent_padded_full_attention_rows": None,
            "persistent_padded_sliding_metadata_padding": False,
            "persistent_padded_decode_cuda_graph": True,
            "persistent_padded_decode_graph_warmup_iters": 1,
            "use_native_gemma_forward": True,
            "enable_token_pool_metadata": None,
            "token_pool_max_context_len": 1024,
            "token_pool_capacity": 4096,
            "token_pool_paged_block_size": None,
            "enable_token_pool_triton": True,
            "enable_token_pool_paged_triton": True,
            "enable_token_pool_paged_split_triton": True,
            "token_pool_triton_strict": True,
            "token_pool_sliding_paged_metadata_only": True,
            "slots": None,
            "token_budget": None,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_native_bench_hf_boundary_summary_uses_row_evidence(self) -> None:
        from experiments.native_gemma_bench import hf_boundary_summary

        args = SimpleNamespace(
            use_native_gemma_forward=False,
            native_gemma_checkpoint_loader=False,
        )
        rows = [
            {
                "B": 2,
                "success_count": 2,
                "model_forward_backend": "wkvm_native_gemma_forward_bridge",
                "uses_hf_transformer_forward": False,
                "uses_hf_model_construction": False,
                "native_gemma_checkpoint_loader": True,
            }
        ]

        summary = hf_boundary_summary(rows, args)

        self.assertEqual(summary["evidence_rows"], 1)
        self.assertEqual(
            summary["model_forward_backend"],
            "wkvm_native_gemma_forward_bridge",
        )
        self.assertFalse(summary["uses_hf_transformer_forward"])
        self.assertFalse(summary["uses_hf_model_construction"])
        self.assertTrue(summary["native_gemma_checkpoint_loader"])

    def test_native_bench_whole_gpu_memory_sets_comparable_green(self) -> None:
        from experiments.native_gemma_bench import finalize_whole_gpu_memory

        class FakeMonitor:
            def __init__(self) -> None:
                self.exited = False

            def __exit__(self, *args) -> None:
                self.exited = True

            def result(self):
                return {
                    "schema": "wkvm.whole_gpu_memory.v1",
                    "scope": "whole_device",
                    "baseline_used_mib": 1024,
                    "peak_used_mib": 19_456,
                    "peak_delta_mib": 18_432,
                }

        monitor = FakeMonitor()
        rows = [{"green": True}]
        result = finalize_whole_gpu_memory(
            monitor,
            rows,
            SimpleNamespace(mem_cap_gib=19.0, headroom_gib=1.0),
        )

        self.assertTrue(monitor.exited)
        self.assertEqual(result["peak_delta_mib"], 18_432)
        self.assertEqual(rows[0]["peak_engine_delta_gib"], 18.0)
        self.assertTrue(rows[0]["torch_reserved_green"])
        self.assertTrue(rows[0]["green"])
        self.assertEqual(
            rows[0]["gpu_memory"]["schema"],
            "wkvm.whole_gpu_memory.v1",
        )

    def test_native_bench_no_hf_requirement_reports_violations(self) -> None:
        from experiments.native_gemma_bench import native_no_hf_requirement_report

        good = {
            "B": 1,
            "success_count": 1,
            "uses_hf_transformer_forward": False,
            "uses_hf_model_construction": False,
            "native_gemma_checkpoint_loader": True,
        }
        bad = {
            "B": 2,
            "success_count": 2,
            "uses_hf_transformer_forward": False,
            "uses_hf_model_construction": True,
            "native_gemma_checkpoint_loader": False,
        }

        self.assertTrue(
            native_no_hf_requirement_report([good], required=True)["passed"]
        )
        report = native_no_hf_requirement_report([good, bad], required=True)
        self.assertFalse(report["passed"])
        self.assertEqual(report["checked_successful_rows"], 2)
        self.assertEqual(
            report["violations"],
            [
                {
                    "B": 2,
                    "problems": [
                        "uses_hf_model_construction_not_false",
                        "native_gemma_checkpoint_loader_not_true",
                    ],
                }
            ],
        )
        empty = native_no_hf_requirement_report([], required=True)
        self.assertFalse(empty["passed"])
        self.assertEqual(
            empty["violations"],
            [{"B": None, "problems": ["no_successful_rows_to_check"]}],
        )
        setup_report = native_no_hf_requirement_report(
            [good],
            required=True,
            setup_problems=["uses_hf_tokenizer_not_false"],
        )
        self.assertFalse(setup_report["passed"])
        self.assertTrue(setup_report["checked_setup_boundary"])
        self.assertEqual(
            setup_report["setup_problems"],
            ["uses_hf_tokenizer_not_false"],
        )
        self.assertEqual(
            setup_report["violations"],
            [
                {
                    "B": None,
                    "phase": "setup",
                    "problems": ["uses_hf_tokenizer_not_false"],
                }
            ],
        )

    def test_native_bench_payload_records_setup_failure_without_rows(self) -> None:
        from experiments.native_gemma_bench import build_benchmark_payload

        args = self.native_bench_payload_args()
        fatal_error = {
            "type": "RuntimeError",
            "message": "CUDA out of memory",
            "phase": "model_load",
        }

        payload = build_benchmark_payload(
            args,
            path="/models/gemma",
            rows=[],
            usable_gib=None,
            token_pool_triton_env={"WKVM_ENABLE_TOKEN_POOL_TRITON": "1"},
            fatal_error=fatal_error,
        )

        self.assertEqual(payload["fatal_error"], fatal_error)
        self.assertEqual(payload["rows"], [])
        self.assertEqual(payload["summary"]["bmax_green"], 0)
        self.assertEqual(payload["torch_usable_gib"], None)
        self.assertEqual(payload["prompt_token_source"], "hf_tokenizer")
        self.assertTrue(payload["uses_hf_tokenizer"])
        self.assertFalse(payload["uses_hf_config"])
        self.assertTrue(payload["native_gemma_config_loader"])
        self.assertFalse(payload["config"]["synthetic_prompts"])
        self.assertEqual(payload["config"]["synthetic_vocab_size"], 262_144)
        self.assertFalse(payload["config"]["uses_hf_config"])
        self.assertTrue(payload["config"]["native_gemma_config_loader"])
        self.assertFalse(payload["uses_hf_transformer_forward"])
        self.assertFalse(payload["uses_hf_model_construction"])
        self.assertTrue(payload["native_gemma_checkpoint_loader"])
        self.assertEqual(payload["hf_boundary"]["evidence_rows"], 0)
        self.assertEqual(
            payload["model_forward_backend"],
            "wkvm_native_gemma_forward_bridge",
        )
        self.assertEqual(
            payload["native_no_hf_requirement"]["violations"],
            [
                {
                    "B": None,
                    "phase": "setup",
                    "problems": ["uses_hf_tokenizer_not_false"],
                },
                {"B": None, "problems": ["no_successful_rows_to_check"]},
            ],
        )
        self.assertFalse(payload["native_no_hf_requirement"]["passed"])

    def test_native_bench_synthetic_tokenizer_avoids_hf_tokenizer(self) -> None:
        from experiments.native_gemma_bench import (
            load_bench_tokenizer,
            prompt_token_source,
            uses_hf_tokenizer,
        )

        class FailingAutoTokenizer:
            @classmethod
            def from_pretrained(cls, path):
                raise AssertionError(f"unexpected HF tokenizer load from {path}")

        old_transformers = sys.modules.get("transformers")
        sys.modules["transformers"] = SimpleNamespace(
            AutoTokenizer=FailingAutoTokenizer,
        )
        try:
            args = self.native_bench_payload_args(
                synthetic_prompts=True,
                synthetic_vocab_size=128,
            )
            tok = load_bench_tokenizer("/unused/model", args)
        finally:
            if old_transformers is None:
                sys.modules.pop("transformers", None)
            else:
                sys.modules["transformers"] = old_transformers

        ids = tok("abcde", add_special_tokens=True).input_ids
        self.assertGreaterEqual(len(ids), 2)
        self.assertEqual(ids[0], tok.bos_token_id)
        self.assertTrue(all(0 <= token_id < 128 for token_id in ids))
        self.assertIsInstance(tok.decode([ids[-1]]), str)
        self.assertEqual(prompt_token_source(args), "synthetic")
        self.assertFalse(uses_hf_tokenizer(args))

    def test_prompt_fingerprint_is_stable_and_order_sensitive(self) -> None:
        from experiments.bench_prompt_utils import (
            SyntheticBenchTokenizer,
            prompt_fingerprint_row_fields,
            prompt_set_fingerprint,
        )

        tok = SyntheticBenchTokenizer(vocab_size=128)
        prompts = [
            tok("alpha", add_special_tokens=True).input_ids,
            tok("beta", add_special_tokens=True).input_ids,
        ]

        first = prompt_set_fingerprint(
            prompts,
            prompt_token_source="synthetic",
        )
        second = prompt_set_fingerprint(
            [list(prompt) for prompt in prompts],
            prompt_token_source="synthetic",
        )
        reversed_fingerprint = prompt_set_fingerprint(
            list(reversed(prompts)),
            prompt_token_source="synthetic",
        )
        fields = prompt_fingerprint_row_fields(first)

        self.assertEqual(first, second)
        self.assertNotEqual(
            first["prompt_token_ids_sha256"],
            reversed_fingerprint["prompt_token_ids_sha256"],
        )
        self.assertEqual(first["prompt_count"], 2)
        self.assertEqual(first["prompt_lengths"], [len(prompt) for prompt in prompts])
        self.assertEqual(fields["prompt_token_source"], "synthetic")
        self.assertEqual(
            fields["prompt_token_ids_sha256"],
            first["prompt_token_ids_sha256"],
        )

    def test_generated_output_fingerprint_is_canonical_and_exact(self) -> None:
        from experiments.bench_prompt_utils import (
            GENERATED_OUTPUT_FINGERPRINT_SCHEMA,
            generated_output_fingerprint,
            generated_output_fingerprint_row_fields,
        )

        outputs = [
            ("bench-2-1", [17, 23]),
            ("bench-2-0", [5, 11, 13]),
        ]
        fingerprint = generated_output_fingerprint(outputs)
        reordered = generated_output_fingerprint(reversed(outputs))
        changed_token = generated_output_fingerprint(
            [("bench-2-1", [17, 29]), ("bench-2-0", [5, 11, 13])]
        )
        changed_request_id = generated_output_fingerprint(
            [("bench-2-2", [17, 23]), ("bench-2-0", [5, 11, 13])]
        )
        fields = generated_output_fingerprint_row_fields(fingerprint)

        self.assertEqual(fingerprint, reordered)
        self.assertEqual(fingerprint["schema"], GENERATED_OUTPUT_FINGERPRINT_SCHEMA)
        self.assertEqual(fingerprint["request_ids"], ["bench-2-0", "bench-2-1"])
        self.assertEqual(fingerprint["output_token_counts"], [3, 2])
        self.assertEqual(fingerprint["output_token_count"], 5)
        self.assertEqual(
            fingerprint["request_output_token_ids_sha256"],
            "7602407e37f25ea06956b1d49d7e467d63063d060332548dc077e0e9d7e457d1",
        )
        self.assertNotEqual(
            fingerprint["request_output_token_ids_sha256"],
            changed_token["request_output_token_ids_sha256"],
        )
        self.assertNotEqual(
            fingerprint["request_output_token_ids_sha256"],
            changed_request_id["request_output_token_ids_sha256"],
        )
        self.assertEqual(fields["generated_output_fingerprint"], fingerprint)
        self.assertEqual(
            fields["request_output_token_ids_sha256"],
            fingerprint["request_output_token_ids_sha256"],
        )

    def test_generated_output_fingerprint_rejects_ambiguous_inputs(self) -> None:
        from experiments.bench_prompt_utils import generated_output_fingerprint

        with self.assertRaisesRegex(ValueError, "request IDs must be unique"):
            generated_output_fingerprint([("request", [1]), ("request", [2])])
        with self.assertRaisesRegex(ValueError, "integers, not bools"):
            generated_output_fingerprint([("request", [True])])
        with self.assertRaisesRegex(ValueError, "must be integers"):
            generated_output_fingerprint([("request", [1.5])])
        with self.assertRaisesRegex(ValueError, "signed 64-bit"):
            generated_output_fingerprint([("request", [1 << 63])])

    def test_native_bench_no_hf_preflight_runs_before_tokenizer_load(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        import experiments.native_gemma_bench as bench

        def fail_tokenizer_load(path, args):
            raise AssertionError(f"unexpected tokenizer load from {path}")

        old_torch_usable_gib = bench.torch_usable_gib
        old_resolve_model_path = bench.resolve_model_path
        old_load_bench_tokenizer = bench.load_bench_tokenizer
        try:
            bench.torch_usable_gib = lambda mem_cap_gib: None
            bench.resolve_model_path = lambda explicit: "/unused/model"
            bench.load_bench_tokenizer = fail_tokenizer_load
            with tempfile.TemporaryDirectory() as raw_tmp:
                payload_path = Path(raw_tmp) / "payload.json"
                args = self.native_bench_payload_args(
                    json=str(payload_path),
                    model_path="/unused/model",
                    synthetic_prompts=False,
                    require_native_no_hf=True,
                    enable_token_pool_triton=False,
                    enable_token_pool_paged_triton=False,
                    enable_token_pool_paged_split_triton=False,
                    token_pool_triton_strict=False,
                    token_pool_sliding_paged_metadata_only=False,
                )

                with self.assertRaisesRegex(
                    RuntimeError,
                    "native no-HF setup requirement failed before tokenizer",
                ):
                    bench.run(args)

                payload = json.loads(payload_path.read_text())
        finally:
            bench.torch_usable_gib = old_torch_usable_gib
            bench.resolve_model_path = old_resolve_model_path
            bench.load_bench_tokenizer = old_load_bench_tokenizer

        self.assertEqual(
            payload["fatal_error"]["phase"],
            "native_no_hf_setup_validation",
        )
        self.assertEqual(
            payload["native_no_hf_requirement"]["setup_problems"],
            ["uses_hf_tokenizer_not_false"],
        )
        self.assertEqual(payload["rows"], [])

    def test_native_bench_payload_records_synthetic_prompt_source(self) -> None:
        from experiments.native_gemma_bench import build_benchmark_payload

        args = self.native_bench_payload_args(
            synthetic_prompts=True,
            synthetic_vocab_size=128,
        )

        payload = build_benchmark_payload(
            args,
            path="/models/gemma",
            rows=[],
            usable_gib=7.5,
            token_pool_triton_env={},
        )

        self.assertEqual(payload["prompt_token_source"], "synthetic")
        self.assertFalse(payload["uses_hf_tokenizer"])
        self.assertFalse(payload["uses_hf_config"])
        self.assertTrue(payload["native_gemma_config_loader"])
        self.assertEqual(payload["native_no_hf_requirement"]["setup_problems"], [])
        self.assertTrue(payload["config"]["synthetic_prompts"])
        self.assertEqual(payload["config"]["synthetic_vocab_size"], 128)
        self.assertFalse(payload["config"]["uses_hf_config"])
        self.assertTrue(payload["config"]["native_gemma_config_loader"])

    def test_native_bench_payload_indexes_output_fingerprints_by_batch(self) -> None:
        from experiments.bench_prompt_utils import (
            GENERATED_OUTPUT_FINGERPRINT_SCHEMA,
            generated_output_fingerprint,
            generated_output_fingerprint_row_fields,
        )
        from experiments.native_gemma_bench import build_benchmark_payload

        args = self.native_bench_payload_args(
            synthetic_prompts=True,
            synthetic_vocab_size=128,
        )
        fingerprint = generated_output_fingerprint(
            [("bench-2-0", [5, 11]), ("bench-2-1", [17, 23])]
        )
        rows = [
            {
                "B": 2,
                "success_count": 2,
                "error_count": 0,
                "green": True,
                **generated_output_fingerprint_row_fields(fingerprint),
            },
            {
                "B": 4,
                "success_count": 3,
                "error_count": 1,
                "green": False,
            },
        ]

        payload = build_benchmark_payload(
            args,
            path="/models/gemma",
            rows=rows,
            usable_gib=7.5,
            token_pool_triton_env={},
        )

        self.assertEqual(
            payload["generated_output_fingerprint_schema"],
            GENERATED_OUTPUT_FINGERPRINT_SCHEMA,
        )
        self.assertEqual(
            payload["generated_output_fingerprint_coverage"],
            {
                "successful_rows": 1,
                "fingerprinted_successful_rows": 1,
                "complete": True,
            },
        )
        self.assertEqual(
            payload["generated_output_fingerprints_by_batch"],
            {"2": fingerprint},
        )

    def test_native_output_fingerprint_coverage_detects_missing_success(self) -> None:
        from experiments.native_gemma_bench import (
            generated_output_fingerprint_summary,
        )

        summary = generated_output_fingerprint_summary(
            [{"B": 2, "success_count": 2, "error_count": 0}]
        )

        self.assertEqual(summary["successful_rows"], 1)
        self.assertEqual(summary["fingerprinted_successful_rows"], 0)
        self.assertFalse(summary["complete"])
        self.assertEqual(summary["by_batch"], {})

    def test_native_bench_payload_records_hf_config_for_hf_loader(self) -> None:
        from experiments.native_gemma_bench import build_benchmark_payload

        args = self.native_bench_payload_args(
            native_gemma_checkpoint_loader=False,
            use_native_gemma_forward=False,
        )

        payload = build_benchmark_payload(
            args,
            path="/models/gemma",
            rows=[],
            usable_gib=7.5,
            token_pool_triton_env={},
        )

        self.assertTrue(payload["uses_hf_config"])
        self.assertFalse(payload["native_gemma_config_loader"])
        self.assertTrue(payload["config"]["uses_hf_config"])
        self.assertFalse(payload["config"]["native_gemma_config_loader"])
        self.assertEqual(
            payload["native_no_hf_requirement"]["setup_problems"],
            [
                "uses_hf_tokenizer_not_false",
                "uses_hf_config_not_false",
                "native_gemma_config_loader_not_true",
            ],
        )

    def test_native_bench_applies_token_pool_triton_env_flags(self) -> None:
        from experiments.native_gemma_bench import (
            TOKEN_POOL_TRITON_BENCH_ENV_NAMES,
            apply_token_pool_triton_bench_env,
        )

        old_env = {
            name: os.environ.get(name)
            for name in TOKEN_POOL_TRITON_BENCH_ENV_NAMES
        }
        try:
            for name in TOKEN_POOL_TRITON_BENCH_ENV_NAMES:
                os.environ.pop(name, None)
            args = SimpleNamespace(
                enable_token_pool_triton=True,
                enable_token_pool_paged_triton=True,
                enable_token_pool_paged_split_triton=False,
                token_pool_triton_strict=True,
                token_pool_sliding_paged_metadata_only=True,
            )

            report = apply_token_pool_triton_bench_env(args)

            self.assertEqual(report["WKVM_ENABLE_TOKEN_POOL_TRITON"], "1")
            self.assertEqual(report["WKVM_ENABLE_TOKEN_POOL_PAGED_TRITON"], "1")
            self.assertIsNone(report["WKVM_ENABLE_TOKEN_POOL_PAGED_SPLIT_TRITON"])
            self.assertEqual(report["WKVM_TOKEN_POOL_TRITON_STRICT"], "1")
            self.assertEqual(
                report["WKVM_TOKEN_POOL_SLIDING_PAGED_METADATA_ONLY"],
                "1",
            )
        finally:
            for name, value in old_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            from wkvm.runner.gemma_token_pool_attention import (
                reset_token_pool_triton_dispatch_plan_cache,
            )

            reset_token_pool_triton_dispatch_plan_cache()

    def test_native_bench_engine_honors_prefill_chunk_cap(self) -> None:
        from experiments.native_gemma_bench import make_engine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
        )
        args = SimpleNamespace(
            slots=None,
            token_budget=None,
            chunk=4,
            decode_microbatch_rows=16,
            decode_microbatch_bytes=None,
            decode_batch_planner="scheduler",
            decode_workspace_bytes=None,
            decode_workspace_width_bucket=16,
            disable_persistent_exact_decode=False,
            disable_persistent_padded_decode=False,
            persistent_padded_decode_steps=8,
            persistent_padded_decode_cuda_graph=False,
            persistent_padded_decode_graph_warmup_iters=3,
            use_native_gemma_forward=False,
            native_gemma_attention_backend="manual",
            native_gemma_projection_backend="separate",
            native_gemma_weight_backend="hf_live",
        )

        engine = make_engine(
            FakeModel(),
            cfg,
            [list(range(10)), list(range(12))],
            args,
        )

        self.assertEqual(engine.scheduler.config.max_tokens_per_request_per_step, 4)

    def test_engine_exposes_persistent_padded_cuda_graph_config(self) -> None:
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sink_tokens=7,
            ring_tokens=33,
            routed_slots=5,
            pending_tokens=11,
            sliding_window=99,
        )
        engine = GemmaNativeEngine(
            model=FakeModel(),
            config=cfg,
            num_slots=1,
            persistent_padded_decode_cuda_graph=True,
            persistent_padded_decode_graph_warmup_iters=0,
        )

        self.assertTrue(engine.persistent_padded_decode_cuda_graph)
        self.assertEqual(engine.persistent_padded_decode_graph_warmup_iters, 0)
        self.assertTrue(engine.runner.persistent_padded_decode_cuda_graph)
        self.assertEqual(engine.runner.persistent_padded_decode_graph_warmup_iters, 0)
        self.assertFalse(engine.collect_cuda_memory_phase_metrics)
        self.assertFalse(engine.runner.collect_cuda_memory_phase_metrics)
        stats = engine.stats()
        self.assertTrue(stats["persistent_padded_decode_cuda_graph"])
        self.assertEqual(stats["persistent_padded_decode_graph_warmup_iters"], 0)
        self.assertFalse(stats["persistent_padded_full_attention_rows"])
        self.assertFalse(stats["persistent_padded_sliding_metadata_padding"])
        self.assertFalse(stats["cuda_phase_metrics_enabled"])
        self.assertFalse(stats["use_native_gemma_forward"])
        self.assertEqual(stats["native_gemma_attention_backend"], "manual")
        self.assertEqual(stats["native_gemma_projection_backend"], "separate")
        self.assertEqual(stats["native_gemma_weight_backend"], "hf_live")
        self.assertFalse(stats["native_gemma_release_hf_decoder_layers"])
        self.assertEqual(stats["native_gemma_released_hf_decoder_layers"], 0)
        self.assertEqual(stats["model_forward_backend"], "hf_transformers_gemma4_forward")
        self.assertTrue(stats["uses_hf_transformer_forward"])
        self.assertEqual(
            stats["native_config"],
            {
                "sink_tokens": 7,
                "ring_tokens": 33,
                "routed_slots": 5,
                "pending_tokens": 11,
                "sliding_window": 99,
            },
        )

    def test_persistent_full_attention_rows_auto_for_token_pool_attention(self) -> None:
        import torch
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        class FakeNativeTokenPoolModel:
            wkvm_no_hf_transformer_forward = True

            def __init__(self) -> None:
                self._param = torch.empty((), dtype=torch.float32)
                self.config = SimpleNamespace(num_attention_heads=2)
                attn = SimpleNamespace(
                    layer_type="sliding_attention",
                    is_kv_shared_layer=False,
                    num_key_value_groups=2,
                    head_dim=4,
                )
                self.text_prefix = SimpleNamespace(
                    layers=[SimpleNamespace(layer_idx=0, attn_meta=attn)]
                )

            def parameters(self):
                return iter([self._param])

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
        )

        auto_engine = GemmaNativeEngine(
            model=FakeNativeTokenPoolModel(),
            config=cfg,
            num_slots=1,
            enable_token_pool_attention=True,
        )
        disabled_engine = GemmaNativeEngine(
            model=FakeNativeTokenPoolModel(),
            config=cfg,
            num_slots=1,
            enable_token_pool_attention=True,
            persistent_padded_full_attention_rows=False,
        )
        no_pool_engine = GemmaNativeEngine(
            model=FakeNativeTokenPoolModel(),
            config=cfg,
            num_slots=1,
        )

        self.assertTrue(auto_engine.persistent_padded_full_attention_rows)
        self.assertTrue(auto_engine.stats()["persistent_padded_full_attention_rows"])
        self.assertFalse(disabled_engine.persistent_padded_full_attention_rows)
        self.assertFalse(no_pool_engine.persistent_padded_full_attention_rows)

    def test_engine_accepts_packed_gate_up_projection_backend(self) -> None:
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
        )
        engine = GemmaNativeEngine(
            model=FakeModel(),
            config=cfg,
            num_slots=1,
            native_gemma_projection_backend="qkv_gate_up_packed",
        )

        stats = engine.stats()
        self.assertEqual(stats["native_gemma_projection_backend"], "qkv_gate_up_packed")
        self.assertEqual(engine.runner.native_gemma_projection_backend, "qkv_gate_up_packed")

    def test_release_hf_decoder_layers_requires_owned_weight_backend(self) -> None:
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
        )
        with self.assertRaisesRegex(ValueError, "requires"):
            GemmaNativeEngine(
                model=FakeModel(),
                config=cfg,
                num_slots=1,
                native_gemma_weight_backend="hf_live",
                native_gemma_release_hf_decoder_layers=True,
            )

    def test_distinct_history_decode_batch_under_shared_scheduler(self) -> None:
        from wkvm.core.arena import StateArena
        from wkvm.core.config import ModelStateSpec, StateFamilySpec
        from wkvm.core.scheduler import Scheduler

        spec = ModelStateSpec(families=(StateFamilySpec("gemma_routed_span", 8),))
        scheduler = Scheduler(
            SchedulerConfig(
                max_tokens_per_step=64,
                max_running_requests=2,
                max_tokens_per_request_per_step=32,
            ),
            StateArena(spec, num_slots=2),
        )
        a = Request(prompt_token_ids=list(range(10)), max_new_tokens=3, req_id="a")
        b = Request(prompt_token_ids=list(range(13)), max_new_tokens=3, req_id="b")
        scheduler.add_request(a)
        scheduler.add_request(b)

        out = scheduler.schedule()
        self.assertEqual(out.num_scheduled_tokens, {"a": 10, "b": 13})
        scheduler.update_from_output(out, {"a": [101], "b": [201]})

        out = scheduler.schedule()
        decode_rows = [scheduler.requests[req_id] for req_id, n in out.num_scheduled_tokens.items() if n == 1]
        self.assertEqual({req.req_id for req in decode_rows}, {"a", "b"})
        self.assertGreater(len({req.num_tokens for req in decode_rows}), 1)


class FakeLogit:
    def __init__(self, token_id: int) -> None:
        self.token_id = token_id

    def argmax(self):
        return self

    def item(self) -> int:
        return self.token_id


class FakeLogits:
    def __init__(self, token_ids: list[int]) -> None:
        self.rows = [FakeLogit(token_id) for token_id in token_ids]

    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            row = idx[0]
            return self.rows[row]
        return self.rows[idx]


class TestGemmaEngineSampling(unittest.TestCase):
    def test_sample_argmax_token_ids_handles_batched_tensor_logits(self) -> None:
        import torch

        logits = torch.tensor(
            [
                [0.1, 0.9, 0.2],
                [0.7, 0.3, 0.1],
            ]
        )

        self.assertEqual(_sample_argmax_token_ids(logits, rows=2), [1, 0])

    def test_sample_argmax_token_ids_handles_sequence_tensor_logits(self) -> None:
        import torch

        logits = torch.tensor(
            [
                [[0.1, 0.2, 0.3], [0.7, 0.4, 0.1]],
                [[0.5, 0.6, 0.1], [0.1, 0.2, 0.9]],
            ]
        )

        self.assertEqual(_sample_argmax_token_ids(logits, rows=2), [0, 2])

    def test_sample_argmax_token_ids_keeps_fake_logits_fallback(self) -> None:
        self.assertEqual(_sample_argmax_token_ids(FakeLogits([12, 34]), rows=2), [12, 34])


class FakeTensorShape:
    dtype = "torch.int32"
    device = "cuda:0"

    def __init__(self, shape: tuple[int, ...]) -> None:
        self.shape = shape

    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= int(dim)
        return total


def fake_token_pool_decode_context(
    *,
    rows: int = 2,
    kv_indices: int = 4,
    layer_id: int = 0,
):
    from wkvm.runner.gemma_token_pool import DecodeBatchMetadata, TokenPoolDecodeContext

    metadata = DecodeBatchMetadata(
        req_pool_indices=FakeTensorShape((rows,)),
        seq_lens=FakeTensorShape((rows,)),
        logical_seq_lens=FakeTensorShape((rows,)),
        out_cache_loc=FakeTensorShape((rows,)),
        kv_indptr=FakeTensorShape((rows + 1,)),
        kv_indices=FakeTensorShape((kv_indices,)),
    )
    return TokenPoolDecodeContext(
        metadata_by_layer_type={"sliding_attention": metadata},
        kv_pool=object(),
        metadata_by_layer_id={layer_id: metadata},
        covered_layer_types=frozenset({"sliding_attention"}),
    )


class FakeBatchRunner:
    def __init__(self) -> None:
        self.prefill_calls: list[list[int]] = []
        self.prefill_chunk_calls: list[tuple[list[int], int]] = []
        self.prefill_chunk_cache_ids: list[int] = []
        self.caches_built = 0
        self.decode_batch_calls: list[tuple[list[int], list[int]]] = []
        self.decode_batch_token_pool_contexts = []
        self.decode_step_calls = 0
        self.decode_step_token_pool_contexts = []
        self.last_decode_batch_info = {"merge": "exact_structural_concat"}

    def build_cache(self, slots):
        self.caches_built += 1
        return FakeCache()

    def prefill(self, token_ids, slots, *, break_mask=None):
        self.prefill_calls.append(list(token_ids))
        return FakeLogit(100 + len(self.prefill_calls)), FakeCache()

    def prefill_chunk_step(self, cache, token_ids, slots, *, start_pos, break_mask=None):
        self.prefill_chunk_calls.append((list(token_ids), int(start_pos)))
        self.prefill_chunk_cache_ids.append(id(cache))
        return FakeLogits([100 + len(self.prefill_chunk_calls)])

    def decode_batch(self, caches, last_tokens, *, position_ids=None, token_pool_decode=None):
        self.decode_batch_calls.append((list(last_tokens), list(position_ids or [])))
        self.decode_batch_token_pool_contexts.append(token_pool_decode)
        return FakeLogits([200 + i for i in range(len(last_tokens))])

    def decode_step(self, cache, last_tokens, *, position_ids=None, token_pool_decode=None):
        self.decode_step_calls += 1
        self.decode_step_token_pool_contexts.append(token_pool_decode)
        return FakeLogits([999])


class FakeFailingPrefillRunner(FakeBatchRunner):
    def prefill(self, token_ids, slots, *, break_mask=None):
        raise RuntimeError("synthetic prefill failure")

    def prefill_chunk_step(self, cache, token_ids, slots, *, start_pos, break_mask=None):
        raise RuntimeError("synthetic prefill failure")


class FakeFailingContinuationPrefillRunner(FakeBatchRunner):
    def prefill_chunk_step(self, cache, token_ids, slots, *, start_pos, break_mask=None):
        if start_pos > 0:
            raise RuntimeError("synthetic continuation prefill failure")
        return super().prefill_chunk_step(
            cache,
            token_ids,
            slots,
            start_pos=start_pos,
            break_mask=break_mask,
        )


class FakeAuthoritativePrefillRunner(FakeBatchRunner):
    def __init__(self, model, hf_config, native_config) -> None:
        super().__init__()
        self.model = model
        self.hf_config = hf_config
        self.native_config = native_config
        self.last_cache = None
        self.last_keys = None
        self.last_values = None

    def build_cache(self, slots):
        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        self.caches_built += 1
        cache = NativeGemmaRoutedCache(self.hf_config, self.native_config)
        self.last_cache = cache
        return cache

    def prefill_chunk_step(self, cache, token_ids, slots, *, start_pos, break_mask=None):
        import torch

        self.prefill_chunk_calls.append((list(token_ids), int(start_pos)))
        self.prefill_chunk_cache_ids.append(id(cache))
        width = len(token_ids)
        start = int(start_pos) * 4
        keys = torch.arange(
            start,
            start + width * 4,
            dtype=torch.float32,
        ).reshape(1, 1, width, 4)
        values = keys + 1000
        returned_k, returned_v = cache.layers[0].update(keys, values)
        cache.store_shared_kv(
            layer_idx=0,
            layer_type="sliding_attention",
            key_states=returned_k,
            value_states=returned_v,
        )
        self.last_keys = keys
        self.last_values = values
        return FakeLogits([100 + len(self.prefill_chunk_calls)])


class FakeSessionTokenPoolRunner(FakeAuthoritativePrefillRunner):
    def __init__(self, model, hf_config, native_config) -> None:
        super().__init__(model, hf_config, native_config)
        self.prefill_entry_tails = []

    def _prefill_cache(self, cache, token_ids, start_pos):
        import torch

        if int(start_pos) > 0:
            layer = cache.layers[0]
            self.prefill_entry_tails.append(
                (
                    id(cache),
                    int(layer.cumulative_length),
                    layer.keys.clone(),
                    layer.values.clone(),
                )
            )
        width = len(token_ids)
        start = int(start_pos) * 4
        keys = torch.arange(
            start,
            start + width * 4,
            dtype=torch.float32,
        ).reshape(1, 1, width, 4)
        values = keys + 1000
        returned_k, returned_v = cache.layers[0].update(keys, values)
        cache.store_shared_kv(
            layer_idx=0,
            layer_type="sliding_attention",
            key_states=returned_k,
            value_states=returned_v,
        )
        self.last_keys = keys
        self.last_values = values

    def prefill_chunk_step(self, cache, token_ids, slots, *, start_pos, break_mask=None):
        self.prefill_chunk_calls.append((list(token_ids), int(start_pos)))
        self.prefill_chunk_cache_ids.append(id(cache))
        self._prefill_cache(cache, token_ids, start_pos)
        return FakeLogits([100 + len(self.prefill_chunk_calls)])

    def prefill_batch_step(
        self,
        caches,
        token_rows,
        slots,
        *,
        start_positions,
        break_masks=None,
    ):
        logits = []
        for cache, token_ids, start_pos in zip(caches, token_rows, start_positions):
            self.prefill_chunk_calls.append((list(token_ids), int(start_pos)))
            self.prefill_chunk_cache_ids.append(id(cache))
            self._prefill_cache(cache, token_ids, start_pos)
            logits.append(100 + len(self.prefill_chunk_calls))
        return FakeLogits(logits)

    def _write_decode_rows(self, token_pool_decode, position_ids):
        import torch

        metadata = token_pool_decode.metadata_for_layer(0, "sliding_attention")
        if metadata is None:
            metadata = token_pool_decode.paged_metadata_for_layer(
                0,
                "sliding_attention",
            )
        if metadata is None:
            raise AssertionError("missing sliding token-pool metadata")
        key_rows = torch.as_tensor(position_ids, dtype=torch.float32).reshape(-1, 1, 1)
        key_rows = key_rows * 4 + torch.arange(4, dtype=torch.float32).reshape(1, 1, 4)
        token_pool_decode.kv_pool.set_kv(
            0,
            metadata.out_cache_loc,
            key_rows,
            key_rows + 1000,
        )

    def decode_step(self, cache, last_tokens, *, position_ids=None, token_pool_decode=None):
        self.decode_step_calls += 1
        self.decode_step_token_pool_contexts.append(token_pool_decode)
        self._write_decode_rows(token_pool_decode, position_ids)
        return FakeLogits([200 + self.decode_step_calls])

    def decode_batch(self, caches, last_tokens, *, position_ids=None, token_pool_decode=None):
        self.decode_batch_calls.append((list(last_tokens), list(position_ids or [])))
        self.decode_batch_token_pool_contexts.append(token_pool_decode)
        self._write_decode_rows(token_pool_decode, position_ids)
        return FakeLogits([300 + row for row in range(len(last_tokens))])


class FakeModel:
    device = "cpu"


class FakeCache:
    def __init__(self) -> None:
        self.split_count = 0

    def state_bytes(self) -> int:
        return 0

    def set_span_break_mask(self, break_mask) -> None:
        self.break_mask = break_mask

    def split_exact_decode_into(self, caches) -> None:
        self.split_count += 1
        for cache in caches:
            cache.restored_from_persistent = True


class FakeRunnerBank:
    def __init__(self) -> None:
        self.ingested: list[tuple[dict[str, int], list[int], list[bool] | None]] = []

    def ingest_positions(self, slots, positions, break_mask=None) -> None:
        self.ingested.append((dict(slots), list(positions), break_mask))


class FakeTorchTensor:
    def __init__(self, data) -> None:
        self.data = data

    def unsqueeze(self, dim: int):
        if dim != 0:
            raise AssertionError(f"unexpected unsqueeze dim {dim}")
        return FakeTorchTensor([self.data])


class FakeTorchModule:
    long = "long"

    def tensor(self, data, *, dtype=None, device=None):
        return FakeTorchTensor(list(data))

    def arange(self, start, end, *, dtype=None, device=None):
        return FakeTorchTensor(list(range(start, end)))

    class inference_mode:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False


class RecordingGemmaModel:
    device = "cpu"
    config = SimpleNamespace()

    def __init__(self) -> None:
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(logits=FakeLogits([321]))


class FakePaddedBatchRunner(FakeBatchRunner):
    def decode_batch(self, caches, last_tokens, *, position_ids=None, token_pool_decode=None):
        self.last_decode_batch_info = {
            "merge": "padded_valid_mask_concat",
            "timing": {
                "merge_s": 0.1,
                "model_forward_s": 0.2,
                "commit_s": 0.03,
                "split_s": 0.0,
                "mask_s": 0.04,
                "total_s": 0.37,
            },
            "layers": [
                {
                    "temporary_total_bytes": 1000,
                    "temporary_mask_bytes": 10,
                    "copied_kv_bytes": 700,
                    "padded_kv_bytes": 200,
                    "source_padded_kv_bytes": 150,
                    "workspace_extra_padded_kv_bytes": 50,
                    "reserved_decode_kv_bytes": 100,
                    "workspace_allocated": 1,
                    "workspace_reused": 0,
                    "workspace_bypassed": 0,
                    "pad_slots_total": 5,
                    "workspace_extra_pad_slots_total": 2,
                },
                {
                    "temporary_total_bytes": 2000,
                    "temporary_mask_bytes": 20,
                    "copied_kv_bytes": 1400,
                    "padded_kv_bytes": 400,
                    "source_padded_kv_bytes": 300,
                    "workspace_extra_padded_kv_bytes": 100,
                    "reserved_decode_kv_bytes": 200,
                    "workspace_allocated": 0,
                    "workspace_reused": 1,
                    "workspace_bypassed": 1,
                    "pad_slots_total": 7,
                    "workspace_extra_pad_slots_total": 3,
                },
            ],
        }
        return super().decode_batch(
            caches,
            last_tokens,
            position_ids=position_ids,
            token_pool_decode=token_pool_decode,
        )


class FakeDistinctBatchRunner(FakeBatchRunner):
    def decode_batch(self, caches, last_tokens, *, position_ids=None, token_pool_decode=None):
        self.decode_batch_calls.append((list(last_tokens), list(position_ids or [])))
        self.decode_batch_token_pool_contexts.append(token_pool_decode)
        from wkvm.runner.gemma_runner import DistinctCacheBatchError

        raise DistinctCacheBatchError("synthetic incompatible cache")


class FakePersistentExactBatchRunner(FakeBatchRunner):
    def __init__(self) -> None:
        super().__init__()
        self.persistent_starts: list[tuple[list[int], list[int]]] = []
        self.persistent_reuses: list[tuple[list[int], list[int]]] = []

    def decode_batch_exact_persistent(self, caches, last_tokens, *, position_ids=None, token_pool_decode=None):
        self.persistent_starts.append((list(last_tokens), list(position_ids or [])))
        merged = caches[0]
        self.last_decode_batch_info = {"merge": "exact_structural_concat"}
        return FakeLogits([300 + i for i in range(len(last_tokens))]), merged

    def decode_persistent_exact_batch(self, merged_cache, last_tokens, *, position_ids=None, token_pool_decode=None):
        self.persistent_reuses.append((list(last_tokens), list(position_ids or [])))
        base = 400 + 10 * len(self.persistent_reuses)
        self.last_decode_batch_info = {
            "merge": "exact_structural_concat",
            "persistent_exact_decode": "reuse",
        }
        return FakeLogits([base + i for i in range(len(last_tokens))])

    def decode_batch(self, caches, last_tokens, *, position_ids=None, token_pool_decode=None):
        raise AssertionError("persistent exact decode should avoid regular decode_batch")


class FakePersistentPaddedCache:
    def __init__(self, reserve_steps: int) -> None:
        self.remaining = reserve_steps
        self.remaining_capacity_calls = 0
        self.commit_count = 0

    def padded_decode_remaining_capacity(self) -> int:
        self.remaining_capacity_calls += 1
        return self.remaining

    def consume(self) -> None:
        if self.remaining < 1:
            raise AssertionError("persistent padded cache over-consumed")
        self.remaining -= 1

    def commit_padded_decode_into(self, caches) -> None:
        self.commit_count += 1
        for cache in caches:
            cache.restored_from_persistent_padded = True


class FakePersistentPaddedBatchRunner(FakeBatchRunner):
    def __init__(self) -> None:
        super().__init__()
        self.persistent_padded_starts: list[tuple[list[int], list[int], int]] = []
        self.persistent_padded_reuses: list[tuple[list[int], list[int]]] = []
        self.merged_cache: FakePersistentPaddedCache | None = None

    def decode_batch_padded_persistent(self, caches, last_tokens, *, position_ids=None, reserve_steps=1, token_pool_decode=None):
        self.persistent_padded_starts.append(
            (list(last_tokens), list(position_ids or []), int(reserve_steps))
        )
        self.merged_cache = FakePersistentPaddedCache(reserve_steps)
        self.merged_cache.consume()
        self.last_decode_batch_info = {
            "merge": "padded_valid_mask_concat",
            "layers": [
                {
                    "temporary_total_bytes": 1000,
                    "temporary_mask_bytes": 10,
                    "copied_kv_bytes": 700,
                    "padded_kv_bytes": 200,
                    "source_padded_kv_bytes": 150,
                    "workspace_extra_padded_kv_bytes": 0,
                    "reserved_decode_kv_bytes": 100,
                    "workspace_allocated": 0,
                    "workspace_reused": 0,
                    "workspace_bypassed": 0,
                    "pad_slots_total": 5,
                    "workspace_extra_pad_slots_total": 0,
                },
            ],
        }
        return FakeLogits([500 + i for i in range(len(last_tokens))]), self.merged_cache

    def decode_persistent_padded_batch(self, merged_cache, last_tokens, *, position_ids=None, token_pool_decode=None):
        self.persistent_padded_reuses.append((list(last_tokens), list(position_ids or [])))
        merged_cache.consume()
        base = 600 + 10 * len(self.persistent_padded_reuses)
        self.last_decode_batch_info = {
            "merge": "padded_valid_mask_concat",
            "persistent_padded_decode": "reuse",
        }
        return FakeLogits([base + i for i in range(len(last_tokens))])

    def decode_batch(self, caches, last_tokens, *, position_ids=None, token_pool_decode=None):
        raise AssertionError("persistent padded decode should avoid regular decode_batch")


class FakeGraphMismatchPersistentPaddedBatchRunner(FakePersistentPaddedBatchRunner):
    def decode_persistent_padded_batch(
        self,
        merged_cache,
        last_tokens,
        *,
        position_ids=None,
        token_pool_decode=None,
    ):
        self.persistent_padded_reuses.append(
            (list(last_tokens), list(position_ids or []))
        )
        from wkvm.runner.gemma_runner import DistinctCacheBatchError

        raise DistinctCacheBatchError(
            "token-pool cuda graph metadata incompatible: "
            "metadata_by_layer_type.sliding_attention.kv_indices"
        )

    def decode_batch(
        self,
        caches,
        last_tokens,
        *,
        position_ids=None,
        token_pool_decode=None,
    ):
        return FakeBatchRunner.decode_batch(
            self,
            caches,
            last_tokens,
            position_ids=position_ids,
            token_pool_decode=token_pool_decode,
        )


class TestGemmaRoutedSpanRunner(unittest.TestCase):
    @staticmethod
    def _graph_cache_context(*, kv_pool, attention_workspace, kv_indices: int = 6):
        from wkvm.runner.gemma_token_pool import (
            DecodeBatchMetadata,
            TokenPoolDecodeContext,
        )

        class FakeCudaTensor:
            is_cuda = True
            dtype = "torch.int32"
            device = "cuda:0"

            def __init__(self, shape) -> None:
                self.shape = tuple(shape)

            def numel(self) -> int:
                total = 1
                for dim in self.shape:
                    total *= int(dim)
                return total

        rows = 2
        metadata = DecodeBatchMetadata(
            req_pool_indices=FakeCudaTensor((rows,)),
            seq_lens=FakeCudaTensor((rows,)),
            logical_seq_lens=FakeCudaTensor((rows,)),
            out_cache_loc=FakeCudaTensor((rows,)),
            kv_indptr=FakeCudaTensor((rows + 1,)),
            kv_indices=FakeCudaTensor((kv_indices,)),
        )
        return TokenPoolDecodeContext(
            metadata_by_layer_type={"sliding_attention": metadata},
            kv_pool=kv_pool,
            attention_workspace=attention_workspace,
            metadata_by_layer_id={0: metadata},
            covered_layer_types=frozenset({"sliding_attention"}),
        )

    @staticmethod
    def _graph_cache_cohort(torch, hf_config, native_config, offset: int):
        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        caches = []
        for row in range(2):
            cache = NativeGemmaRoutedCache(hf_config, native_config)
            key = torch.full((1, 1, 3, 2), float(offset + row + 1))
            value = torch.full((1, 1, 3, 2), float(offset + row + 11))
            cache.update(key, value, layer_idx=0)
            caches.append(cache)
        return caches

    def test_prefill_chunk_step_passes_explicit_position_ids(self) -> None:
        from wkvm.runner.gemma_runner import GemmaRoutedSpanRunner

        previous_torch = sys.modules.get("torch")
        had_torch = "torch" in sys.modules
        sys.modules["torch"] = FakeTorchModule()  # type: ignore[assignment]
        try:
            model = RecordingGemmaModel()
            bank = FakeRunnerBank()
            runner = GemmaRoutedSpanRunner(model, bank, prefill_chunk=2)
            cache = FakeCache()
            slots = {"gemma_routed_span": 7}
            break_mask = [False, True, False]

            logits = runner.prefill_chunk_step(
                cache,
                [11, 12, 13],
                slots,
                start_pos=4,
                break_mask=break_mask,
            )
        finally:
            if had_torch:
                sys.modules["torch"] = previous_torch  # type: ignore[assignment]
            else:
                sys.modules.pop("torch", None)

        self.assertIsInstance(logits, FakeLogits)
        self.assertEqual(len(model.calls), 1)
        call = model.calls[0]
        self.assertEqual(call["input_ids"].data, [[11, 12, 13]])
        self.assertEqual(call["position_ids"].data, [[4, 5, 6]])
        self.assertIs(call["past_key_values"], cache)
        self.assertIs(call["use_cache"], True)
        self.assertEqual(call["logits_to_keep"], 1)
        self.assertEqual(bank.ingested, [(slots, [4, 5, 6], break_mask)])
        self.assertIs(cache.break_mask, break_mask)

    def test_token_pool_decode_mask_adapter_drops_covered_masks(self) -> None:
        from wkvm.runner.gemma_runner import _attention_mask_for_token_pool_decode

        full_mask = object()
        sliding_mask = object()
        mask = {"full_attention": full_mask, "sliding_attention": sliding_mask}
        context = SimpleNamespace(
            metadata_by_layer_type={
                "sliding_attention": SimpleNamespace(out_cache_loc=object())
            },
        )

        adjusted = _attention_mask_for_token_pool_decode(mask, context)

        self.assertIsNot(adjusted, mask)
        self.assertIs(adjusted["full_attention"], full_mask)
        self.assertIsNone(adjusted["sliding_attention"])
        self.assertIs(mask["sliding_attention"], sliding_mask)
        without_sliding_metadata = SimpleNamespace(
            metadata_by_layer_type={"full_attention": object()},
        )
        self.assertIs(
            _attention_mask_for_token_pool_decode(mask, without_sliding_metadata),
            mask,
        )
        full_context = SimpleNamespace(
            metadata_by_layer_type={
                "full_attention": SimpleNamespace(out_cache_loc=object())
            },
        )
        full_adjusted = _attention_mask_for_token_pool_decode(mask, full_context)
        self.assertIsNot(full_adjusted, mask)
        self.assertIsNone(full_adjusted["full_attention"])
        self.assertIs(full_adjusted["sliding_attention"], sliding_mask)

    def test_padded_decode_native_token_pool_context_drops_sliding_mask(self) -> None:
        import torch
        from wkvm.runner.gemma_runner import GemmaRoutedSpanRunner

        class RecordingNativeModel:
            device = "cpu"
            config = SimpleNamespace()
            wkvm_no_hf_transformer_forward = True

            def __init__(self) -> None:
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                batch = int(kwargs["input_ids"].shape[0])
                return SimpleNamespace(logits=torch.zeros(batch, 1, 3))

        class RecordingPaddedCache:
            def __init__(self) -> None:
                self.full_mask = object()
                self.sliding_mask = object()

            def padded_attention_mask(self):
                return {
                    "full_attention": self.full_mask,
                    "sliding_attention": self.sliding_mask,
                }

        model = RecordingNativeModel()
        cache = RecordingPaddedCache()
        context = SimpleNamespace(
            metadata_by_layer_type={
                "sliding_attention": SimpleNamespace(out_cache_loc=object())
            },
        )
        runner = GemmaRoutedSpanRunner(model, FakeRunnerBank())

        logits = runner._decode_padded_cache(
            cache,
            [11, 12],
            position_ids=[4, 5],
            token_pool_decode=context,
        )

        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertEqual(len(model.calls), 1)
        call = model.calls[0]
        self.assertIs(call["wkvm_token_pool_decode"], context)
        self.assertIs(call["attention_mask"]["full_attention"], cache.full_mask)
        self.assertIsNone(call["attention_mask"]["sliding_attention"])

    def test_padded_decode_hf_forward_keeps_sliding_mask_with_context(self) -> None:
        import torch
        from wkvm.runner.gemma_runner import GemmaRoutedSpanRunner

        class RecordingHFModel:
            device = "cpu"
            config = SimpleNamespace()

            def __init__(self) -> None:
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                batch = int(kwargs["input_ids"].shape[0])
                return SimpleNamespace(logits=torch.zeros(batch, 1, 3))

        class RecordingPaddedCache:
            def __init__(self) -> None:
                self.full_mask = object()
                self.sliding_mask = object()

            def padded_attention_mask(self):
                return {
                    "full_attention": self.full_mask,
                    "sliding_attention": self.sliding_mask,
                }

        model = RecordingHFModel()
        cache = RecordingPaddedCache()
        context = SimpleNamespace(
            metadata_by_layer_type={"sliding_attention": object()},
        )
        runner = GemmaRoutedSpanRunner(model, FakeRunnerBank())

        runner._decode_padded_cache(
            cache,
            [11, 12],
            position_ids=[4, 5],
            token_pool_decode=context,
        )

        self.assertEqual(len(model.calls), 1)
        call = model.calls[0]
        self.assertNotIn("wkvm_token_pool_decode", call)
        self.assertIs(call["attention_mask"]["full_attention"], cache.full_mask)
        self.assertIs(call["attention_mask"]["sliding_attention"], cache.sliding_mask)

    def test_persistent_padded_graph_skips_partial_token_pool_coverage(self) -> None:
        import torch
        from unittest.mock import patch

        from wkvm.models.gemma import gemma4_e4b_routed_span_config
        from wkvm.runner.gemma_runner import GemmaRoutedSpanRunner, NativeGemmaRoutedCache

        hf_config = SimpleNamespace(
            num_hidden_layers=2,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention", "full_attention"),
            sliding_window=8,
        )
        native_config = gemma4_e4b_routed_span_config(
            num_hidden_layers=2,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention", "full_attention"),
            sink_tokens=1,
            ring_tokens=8,
            pending_tokens=8,
            routed_slots=2,
            reps_per_slot=1,
            span_budget_tokens=4,
            max_span_tokens=3,
            sliding_window=8,
        )

        class RecordingNativeModel:
            device = "cpu"
            config = hf_config
            wkvm_no_hf_transformer_forward = True

            def __init__(self) -> None:
                self.calls = []

            def __call__(self, **kwargs):
                self.calls.append(kwargs)
                batch = int(kwargs["input_ids"].shape[0])
                return SimpleNamespace(logits=torch.zeros(batch, 1, 3))

        caches = []
        for row in range(2):
            cache = NativeGemmaRoutedCache(hf_config, native_config)
            for layer_idx in range(2):
                key = torch.full((1, 1, 3, 2), float(10 * layer_idx + row + 1))
                cache.update(key, key + 100, layer_idx=layer_idx)
            caches.append(cache)

        model = RecordingNativeModel()
        runner = GemmaRoutedSpanRunner(
            model,
            FakeRunnerBank(),
            persistent_padded_decode_cuda_graph=True,
        )
        runner._can_cuda_graph_decode = lambda: True
        context = self._graph_cache_context(
            kv_pool=object(),
            attention_workspace=object(),
        )

        with patch("wkvm.runner.gemma_runner._GraphedPaddedDecodeStep") as graph_step:
            _logits, merged_cache = runner.decode_batch_padded_persistent(
                caches,
                [11, 12],
                position_ids=[3, 3],
                reserve_steps=2,
                token_pool_decode=context,
            )

        graph_step.assert_not_called()
        self.assertFalse(hasattr(merged_cache, "_padded_decode_graph"))
        self.assertFalse(merged_cache.static_padded_decode)
        self.assertFalse(runner._token_pool_decode_graph_cache)
        self.assertEqual(len(model.calls), 1)
        self.assertIs(model.calls[0]["wkvm_token_pool_decode"], context)
        self.assertIsNone(model.calls[0]["attention_mask"]["sliding_attention"])
        self.assertIsNotNone(model.calls[0]["attention_mask"]["full_attention"])
        self.assertEqual(
            runner.last_decode_batch_info["persistent_padded_decode_cuda_graph_skip"],
            "partial_token_pool_coverage",
        )
        self.assertEqual(
            runner.last_decode_batch_info["persistent_padded_decode_cuda_graph"],
            0,
        )
        self.assertNotIn(
            "persistent_padded_decode_cuda_graph_captured",
            runner.last_decode_batch_info,
        )
        self.assertNotIn("cuda_graph_replay", runner.last_decode_batch_info)

    def test_token_pool_graph_cache_reuses_graph_across_request_cohorts(self) -> None:
        import torch
        from unittest.mock import patch

        from wkvm.models.gemma import gemma4_e4b_routed_span_config
        from wkvm.runner.gemma_runner import GemmaRoutedSpanRunner

        hf_config = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=16,
        )
        native_config = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=16,
        )
        model = SimpleNamespace(
            device="cpu",
            config=hf_config,
            wkvm_no_hf_transformer_forward=True,
        )
        runner = GemmaRoutedSpanRunner(
            model,
            FakeRunnerBank(),
            persistent_padded_decode_cuda_graph=True,
        )
        runner._can_cuda_graph_decode = lambda: True
        seen_graphs = []

        def fake_decode(cache, last_tokens, *, position_ids=None, token_pool_decode=None):
            seen_graphs.append(cache._padded_decode_graph)
            runner.last_decode_batch_info = {"cuda_graph_replay": 1}
            return torch.zeros(len(last_tokens), 3)

        runner._decode_padded_cache = fake_decode
        created = []

        class FakeGraph:
            def __init__(self, model, cache, batch_size, **kwargs) -> None:
                self.model = model
                self.cache = cache
                self.batch_size = batch_size
                created.append(self)

        kv_pool = object()
        attention_workspace = object()
        first_context = self._graph_cache_context(
            kv_pool=kv_pool,
            attention_workspace=attention_workspace,
        )
        second_context = self._graph_cache_context(
            kv_pool=kv_pool,
            attention_workspace=attention_workspace,
        )

        with patch("wkvm.runner.gemma_runner._GraphedPaddedDecodeStep", FakeGraph):
            _first_logits, first_cache = runner.decode_batch_padded_persistent(
                self._graph_cache_cohort(torch, hf_config, native_config, 0),
                [11, 12],
                position_ids=[3, 3],
                reserve_steps=4,
                token_pool_decode=first_context,
            )
            first_info = dict(runner.last_decode_batch_info)
            _second_logits, second_cache = runner.decode_batch_padded_persistent(
                self._graph_cache_cohort(torch, hf_config, native_config, 100),
                [21, 22],
                position_ids=[7, 7],
                reserve_steps=4,
                token_pool_decode=second_context,
            )
            second_info = dict(runner.last_decode_batch_info)

        self.assertEqual(len(created), 1)
        self.assertIsNot(first_cache, second_cache)
        self.assertIs(seen_graphs[0], seen_graphs[1])
        self.assertEqual(first_info["persistent_padded_decode_cuda_graph_captured"], 1)
        self.assertEqual(first_info["persistent_padded_decode_cuda_graph_cache_hit"], 0)
        self.assertEqual(second_info["persistent_padded_decode_cuda_graph_captured"], 0)
        self.assertEqual(second_info["persistent_padded_decode_cuda_graph_cache_hit"], 1)

    def test_dense_padded_decode_skips_cuda_graphs(self) -> None:
        import torch
        from unittest.mock import patch

        from wkvm.models.gemma import gemma4_e4b_routed_span_config
        from wkvm.runner.gemma_runner import GemmaRoutedSpanRunner

        hf_config = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=16,
        )
        native_config = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=16,
        )
        model = SimpleNamespace(device="cpu", config=hf_config)
        runner = GemmaRoutedSpanRunner(
            model,
            FakeRunnerBank(),
            persistent_padded_decode_cuda_graph=True,
        )
        runner._can_cuda_graph_decode = lambda: True

        eager_calls = []

        def fake_decode(cache, last_tokens, *, position_ids=None, token_pool_decode=None):
            eager_calls.append(cache)
            runner.last_decode_batch_info = {}
            return torch.zeros(len(last_tokens), 3)

        runner._decode_padded_cache = fake_decode
        created = []

        class FakeGraph:
            def __init__(self, model, cache, batch_size, **kwargs) -> None:
                created.append(self)

        with patch("wkvm.runner.gemma_runner._GraphedPaddedDecodeStep", FakeGraph):
            runner.decode_batch_padded_persistent(
                self._graph_cache_cohort(torch, hf_config, native_config, 0),
                [11, 12],
                position_ids=[3, 3],
                reserve_steps=4,
            )
            runner.decode_batch_padded_persistent(
                self._graph_cache_cohort(torch, hf_config, native_config, 100),
                [21, 22],
                position_ids=[7, 7],
                reserve_steps=4,
            )

        self.assertEqual(len(created), 0)
        self.assertEqual(len(eager_calls), 2)
        self.assertEqual(
            runner.last_decode_batch_info["persistent_padded_decode_cuda_graph_skip"],
            "unavailable",
        )
        self.assertFalse(runner._token_pool_decode_graph_cache)

    def test_graph_replay_bookkeeping_updates_current_cohort_cache(self) -> None:
        import torch

        from wkvm.runner.gemma_runner import GemmaRoutedSpanRunner

        model = SimpleNamespace(device="cpu", config=SimpleNamespace())
        runner = GemmaRoutedSpanRunner(model, FakeRunnerBank())

        class RecordingCache:
            def __init__(self) -> None:
                self.static_replays = 0
                self.token_pool_steps = 0

            def get_seq_length(self) -> int:
                return 17

            def record_static_padded_decode_replay(self) -> None:
                self.static_replays += 1

            def record_token_pool_covered_decode_step(self) -> None:
                self.token_pool_steps += 1

        original_cache = RecordingCache()
        current_cache = RecordingCache()

        class FakeGraph:
            records_token_pool_decode_steps = True
            last_decode_info = {}

            def __init__(self) -> None:
                self.position_ids = None

            def decode(self, last_tokens, *, position_ids=None, token_pool_decode=None):
                self.position_ids = position_ids
                return torch.zeros(len(last_tokens), 3)

        graph = FakeGraph()
        graph.cache = original_cache
        current_cache._padded_decode_graph = graph

        runner._decode_padded_cache(
            current_cache,
            [11, 12],
            token_pool_decode=object(),
        )

        self.assertEqual(current_cache.static_replays, 1)
        self.assertEqual(current_cache.token_pool_steps, 1)
        self.assertEqual(original_cache.static_replays, 0)
        self.assertEqual(original_cache.token_pool_steps, 0)
        self.assertEqual(graph.position_ids, [17, 17])

    def test_token_pool_graph_cache_is_bounded_lru(self) -> None:
        from wkvm.runner.gemma_runner import (
            GemmaRoutedSpanRunner,
            _TOKEN_POOL_DECODE_GRAPH_CACHE_MAX_ENTRIES,
        )

        runner = GemmaRoutedSpanRunner(
            SimpleNamespace(device="cpu", config=SimpleNamespace()),
            FakeRunnerBank(),
        )
        graphs = [object() for _ in range(_TOKEN_POOL_DECODE_GRAPH_CACHE_MAX_ENTRIES + 2)]
        for index, graph in enumerate(graphs[:-1]):
            runner._cache_token_pool_decode_graph((index,), graph)

        self.assertEqual(
            len(runner._token_pool_decode_graph_cache),
            _TOKEN_POOL_DECODE_GRAPH_CACHE_MAX_ENTRIES,
        )
        self.assertNotIn((0,), runner._token_pool_decode_graph_cache)
        self.assertIs(runner._cached_token_pool_decode_graph((1,)), graphs[1])

        runner._cache_token_pool_decode_graph(
            (_TOKEN_POOL_DECODE_GRAPH_CACHE_MAX_ENTRIES + 1,),
            graphs[-1],
        )

        self.assertIn((1,), runner._token_pool_decode_graph_cache)
        self.assertNotIn((2,), runner._token_pool_decode_graph_cache)

    def test_token_pool_graph_cache_can_be_cleared_at_cohort_boundary(self) -> None:
        from wkvm.runner.gemma_runner import GemmaRoutedSpanRunner

        runner = GemmaRoutedSpanRunner(
            SimpleNamespace(device="cpu", config=SimpleNamespace()),
            FakeRunnerBank(),
        )
        runner._cache_token_pool_decode_graph((1,), object())
        runner._cache_token_pool_decode_graph((2,), object())

        evicted = runner.clear_token_pool_decode_graph_cache()

        self.assertEqual(evicted, 2)
        self.assertFalse(runner._token_pool_decode_graph_cache)
        self.assertEqual(runner.clear_token_pool_decode_graph_cache(), 0)


class FakeTensorShape:
    def __init__(
        self,
        shape: tuple[int, int, int, int],
        *,
        elem_size: int = 2,
    ) -> None:
        self.shape = shape
        self._elem_size = elem_size

    def element_size(self) -> int:
        return self._elem_size


class FakeLayerCache:
    def __init__(self, length: int) -> None:
        self.keys = FakeTensorShape((1, 2, length, 4))
        self.values = FakeTensorShape((1, 2, length, 4))


class FakeSizedCache:
    def __init__(self, lengths: list[int], *, state_bytes: int = 0) -> None:
        self.layers = [FakeLayerCache(length) for length in lengths]
        self._state_bytes = state_bytes

    def state_bytes(self) -> int:
        return self._state_bytes


class TestGemmaNativeEngineDecodeBatch(unittest.TestCase):
    def _make_sliding_session_token_pool_engine(
        self,
        *,
        num_slots: int,
        prefill_microbatch_rows: int,
    ):
        import torch
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        class FakeNativeTokenPoolModel:
            wkvm_no_hf_transformer_forward = True

            def __init__(self) -> None:
                self._param = torch.empty((), dtype=torch.float32)
                self.config = SimpleNamespace(num_attention_heads=2)
                attn = SimpleNamespace(
                    layer_type="sliding_attention",
                    is_kv_shared_layer=False,
                    num_key_value_groups=2,
                    head_dim=4,
                )
                self.text_prefix = SimpleNamespace(
                    layers=[SimpleNamespace(layer_idx=0, attn_meta=attn)]
                )

            def parameters(self):
                return iter([self._param])

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            num_kv_heads=1,
            head_dim=4,
            sliding_window=4,
        )
        model = FakeNativeTokenPoolModel()
        engine = GemmaNativeEngine(
            model=model,
            config=cfg,
            num_slots=num_slots,
            scheduler_config=SchedulerConfig(
                max_tokens_per_step=64,
                max_running_requests=num_slots,
                max_tokens_per_request_per_step=64,
            ),
            prefill_microbatch_rows=prefill_microbatch_rows,
            persistent_exact_decode=False,
            persistent_padded_decode=False,
            enable_token_pool_attention=True,
            token_pool_max_context_len=32,
            token_pool_capacity=128,
        )
        hf_config = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=4,
        )
        runner = FakeSessionTokenPoolRunner(model, hf_config, cfg)
        engine.runner = runner
        return engine, runner

    def _make_full_attention_token_pool_fixture(
        self,
        *,
        req_id: str = "full",
        prompt_token_ids: list[int] | None = None,
        num_slots: int = 1,
        token_pool_capacity: int = 64,
        token_pool_paged_block_size: int | None = None,
    ):
        import torch
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config
        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        class FakeNativeTokenPoolModel:
            wkvm_no_hf_transformer_forward = True

            def __init__(self) -> None:
                self._param = torch.empty((), dtype=torch.float32)
                self.config = SimpleNamespace(
                    num_attention_heads=2,
                    layer_types=("sliding_attention", "full_attention"),
                )
                metas = [
                    SimpleNamespace(
                        layer_type="sliding_attention",
                        is_kv_shared_layer=False,
                        num_key_value_groups=2,
                        head_dim=2,
                    ),
                    SimpleNamespace(
                        layer_type="full_attention",
                        is_kv_shared_layer=False,
                        num_key_value_groups=2,
                        head_dim=2,
                    ),
                ]
                self.text_prefix = SimpleNamespace(
                    layers=[
                        SimpleNamespace(layer_idx=idx, attn_meta=meta)
                        for idx, meta in enumerate(metas)
                    ]
                )

            def parameters(self):
                return iter([self._param])

        hf_cfg = SimpleNamespace(
            num_hidden_layers=2,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention", "full_attention"),
            sliding_window=8,
        )
        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=2,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention", "full_attention"),
            sink_tokens=1,
            ring_tokens=8,
            pending_tokens=8,
            routed_slots=2,
            reps_per_slot=1,
            span_budget_tokens=4,
            max_span_tokens=3,
            sliding_window=8,
        )
        engine = GemmaNativeEngine(
            model=FakeNativeTokenPoolModel(),
            config=cfg,
            num_slots=num_slots,
            enable_token_pool_attention=True,
            token_pool_max_context_len=8,
            token_pool_capacity=token_pool_capacity,
            token_pool_paged_block_size=token_pool_paged_block_size,
        )
        tokens = list(prompt_token_ids or [1, 2, 3])
        req = Request(prompt_token_ids=tokens, max_new_tokens=4, req_id=req_id)
        cache = NativeGemmaRoutedCache(hf_cfg, cfg)
        hidden = len(tokens) * 2
        sliding_keys = torch.arange(hidden, dtype=torch.float32).reshape(
            1, 1, len(tokens), 2
        )
        full_keys = torch.arange(20, 20 + hidden, dtype=torch.float32).reshape(
            1, 1, len(tokens), 2
        )
        cache.update(sliding_keys, sliding_keys + 100, layer_idx=0)
        cache.update(full_keys, full_keys + 100, layer_idx=1)
        engine._caches[req.req_id] = cache
        engine._token_pool_commit_prefill_tokens(
            req,
            req.num_prompt_tokens,
            cache=cache,
            final_prefill=True,
        )
        req.num_computed_tokens = req.num_prompt_tokens
        req.output_token_ids.append(77)
        return engine, req, cache

    def test_token_pool_stats_are_exposed(self) -> None:
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
        )
        engine = GemmaNativeEngine(
            model=FakeModel(),
            config=cfg,
            num_slots=2,
            token_pool_max_context_len=4,
            token_pool_capacity=8,
        )

        stats = engine.stats()
        self.assertTrue(stats["token_pool_metadata_enabled"])
        self.assertEqual(
            stats["token_pool"],
            {
                "enabled": True,
                "attention_enabled": False,
                "active_request_slots": 0,
                "allocated_token_slots": 0,
                "free_token_slots": 0,
                "next_token_slot": 0,
                "token_slot_high_watermark": 0,
                "token_slot_capacity": 8,
                "paged_block_size": 16,
                "page_table_metadata_max_rows": 2,
                "max_context_len": 4,
                "metadata_bytes": 48,
                "kv_pool_bytes": 0,
                "kv_pool_layers": 0,
            },
        )

    def test_token_pool_metadata_defaults_off_without_token_pool_options(self) -> None:
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
        )
        engine = GemmaNativeEngine(
            model=FakeModel(),
            config=cfg,
            num_slots=2,
        )

        stats = engine.stats()
        self.assertFalse(stats["token_pool_metadata_enabled"])
        self.assertEqual(stats["token_pool"], {"enabled": False})

    def test_active_cache_bytes_includes_persistent_groups_once(self) -> None:
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
        )
        engine = GemmaNativeEngine(
            model=FakeModel(),
            config=cfg,
            num_slots=1,
        )
        shared = FakeSizedCache([3], state_bytes=11)
        exact = FakeSizedCache([4], state_bytes=13)
        padded = FakeSizedCache([5], state_bytes=17)
        engine._caches = {"a": shared, "b": exact}
        engine._persistent_exact_decode_groups = {("b",): exact}
        engine._persistent_padded_decode_groups = {("a", "b"): padded}

        self.assertEqual(engine.stats()["active_cache_bytes"], 41)
        engine._record_cache_bytes()
        self.assertEqual(engine.metrics.max_active_cache_bytes, 41)

    def test_cuda_memory_phase_recorder_tracks_current_and_peak_advances(self) -> None:
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
        )
        engine = GemmaNativeEngine(
            model=FakeModel(),
            config=cfg,
            num_slots=1,
            collect_cuda_memory_phase_metrics=True,
        )
        engine.metrics = GemmaEngineMetrics()
        samples = [
            {
                "allocated_bytes": 100,
                "reserved_bytes": 200,
                "max_allocated_bytes": 100,
                "max_reserved_bytes": 200,
            },
            {
                "allocated_bytes": 120,
                "reserved_bytes": 180,
                "max_allocated_bytes": 150,
                "max_reserved_bytes": 200,
            },
            {
                "allocated_bytes": 90,
                "reserved_bytes": 260,
                "max_allocated_bytes": 150,
                "max_reserved_bytes": 260,
            },
        ]
        engine._gpu_memory_stats = lambda: samples.pop(0)  # type: ignore[method-assign]

        engine._record_cuda_memory_phase("prefill_forward")
        engine._record_cuda_memory_phase("decode_model_batch")
        engine._record_cuda_memory_phase("decode_model_batch")

        self.assertEqual(engine.metrics.max_cuda_allocated_bytes, 150)
        self.assertEqual(engine.metrics.max_cuda_allocated_phase, "decode_model_batch")
        self.assertEqual(engine.metrics.max_cuda_reserved_bytes, 260)
        self.assertEqual(engine.metrics.max_cuda_reserved_phase, "decode_model_batch")
        self.assertEqual(
            engine.metrics.cuda_current_reserved_by_phase,
            {"prefill_forward": 200, "decode_model_batch": 260},
        )
        self.assertEqual(
            engine.metrics.cuda_peak_reserved_advances_by_phase,
            {"prefill_forward": 200, "decode_model_batch": 260},
        )

    def test_decode_cuda_memory_info_tracks_runner_subphase_peaks(self) -> None:
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
        )
        engine = GemmaNativeEngine(
            model=FakeModel(),
            config=cfg,
            num_slots=1,
        )
        engine.metrics = GemmaEngineMetrics()

        engine._record_decode_timing_info(
            {
                "persistent_padded_decode_cuda_graph_captured": 1,
                "persistent_padded_decode_cuda_graph_cache_hit": 1,
                "cuda_graph_replay": 1,
                "cuda_memory": {
                    "after_padded_attention_mask": {
                        "allocated_bytes": 100,
                        "reserved_bytes": 200,
                        "max_allocated_bytes": 120,
                        "max_reserved_bytes": 220,
                    },
                    "after_padded_model_forward": {
                        "allocated_bytes": 140,
                        "reserved_bytes": 260,
                        "max_allocated_bytes": 180,
                        "max_reserved_bytes": 300,
                    },
                }
            }
        )

        self.assertEqual(engine.metrics.max_decode_cuda_allocated_bytes, 180)
        self.assertEqual(
            engine.metrics.max_decode_cuda_allocated_phase,
            "after_padded_model_forward",
        )
        self.assertEqual(engine.metrics.max_decode_cuda_reserved_bytes, 300)
        self.assertEqual(
            engine.metrics.max_decode_cuda_reserved_phase,
            "after_padded_model_forward",
        )
        self.assertEqual(
            engine.metrics.decode_cuda_current_reserved_by_phase,
            {"after_padded_attention_mask": 200, "after_padded_model_forward": 260},
        )
        self.assertEqual(
            engine.metrics.decode_cuda_peak_reserved_advances_by_phase,
            {"after_padded_attention_mask": 220, "after_padded_model_forward": 300},
        )
        self.assertEqual(engine.metrics.persistent_padded_decode_cuda_graph_captures, 1)
        self.assertEqual(engine.metrics.persistent_padded_decode_cuda_graph_cache_hits, 1)
        self.assertEqual(engine.metrics.persistent_padded_decode_cuda_graph_replays, 1)

    def test_chunked_prefill_advances_existing_cache_until_gap_closes(self) -> None:
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
        )
        engine = GemmaNativeEngine(
            model=FakeModel(),
            config=cfg,
            num_slots=1,
            enable_token_pool_metadata=True,
            scheduler_config=SchedulerConfig(
                max_tokens_per_step=4,
                max_running_requests=1,
                max_tokens_per_request_per_step=4,
            ),
        )
        runner = FakeBatchRunner()
        engine.runner = runner  # type: ignore[assignment]
        req = Request(prompt_token_ids=list(range(10)), max_new_tokens=2, req_id="chunked")
        engine.add_request(req)

        engine.step()
        self.assertEqual(req.num_computed_tokens, 4)
        self.assertEqual(req.output_token_ids, [])
        self.assertEqual(set(engine._caches), {"chunked"})
        self.assertEqual(engine.stats()["token_pool"]["allocated_token_slots"], 4)

        engine.step()
        self.assertEqual(req.num_computed_tokens, 8)
        self.assertEqual(req.output_token_ids, [])
        self.assertEqual(engine.stats()["token_pool"]["allocated_token_slots"], 8)

        engine.step()
        self.assertEqual(req.num_computed_tokens, 10)
        self.assertEqual(req.output_token_ids, [103])
        self.assertEqual(
            runner.prefill_chunk_calls,
            [
                ([0, 1, 2, 3], 0),
                ([4, 5, 6, 7], 4),
                ([8, 9], 8),
            ],
        )
        self.assertEqual(runner.caches_built, 1)
        self.assertEqual(len(set(runner.prefill_chunk_cache_ids)), 1)
        self.assertEqual(engine.metrics.prefill_calls, 3)
        trace = engine.stats()["requests"]["chunked"]
        self.assertIsNotNone(trace["prefill_time_s"])
        self.assertIsNotNone(trace["first_token_latency_s"])

        engine.step()
        self.assertEqual(runner.decode_batch_calls, [])
        self.assertEqual(runner.decode_step_calls, 1)
        self.assertEqual(req.output_token_ids, [103, 999])
        self.assertIs(req.status, RequestStatus.FINISHED_LENGTH)
        token_pool = engine.stats()["token_pool"]
        self.assertEqual(token_pool["active_request_slots"], 0)
        self.assertEqual(token_pool["allocated_token_slots"], 0)
        self.assertEqual(token_pool["token_slot_high_watermark"], 11)

    def test_session_turn_parks_and_reuses_cache_for_delta_prefill(self) -> None:
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
        )
        engine = GemmaNativeEngine(
            model=FakeModel(),
            config=cfg,
            num_slots=1,
            enable_token_pool_metadata=True,
            scheduler_config=SchedulerConfig(
                max_tokens_per_step=8,
                max_running_requests=1,
                max_tokens_per_request_per_step=8,
            ),
        )
        runner = FakeBatchRunner()
        engine.runner = runner  # type: ignore[assignment]
        req = Request(
            prompt_token_ids=[1, 2, 3],
            max_new_tokens=2,
            req_id="session",
        )
        engine.add_session_request(req, break_mask=[False, False, False])

        engine.step()
        retained_slots = dict(req.slots)
        retained_cache = engine._caches[req.req_id]
        completed = engine.step()
        self.assertEqual(completed, [req])
        self.assertIs(req.status, RequestStatus.PARKED)
        self.assertEqual(req.output_token_ids, [101, 999])
        self.assertEqual(engine.stats()["resident_sessions"], 1)
        self.assertEqual(engine.stats()["parked_sessions"], 1)
        self.assertEqual(engine.stats()["token_pool"]["active_request_slots"], 1)
        first_trace = engine.finished_traces[req.req_id].as_dict()
        self.assertEqual(first_trace["turn_index"], 0)
        self.assertEqual(first_trace["finish_reason"], "length")

        with self.assertRaisesRegex(ValueError, "changed retained history"):
            engine.continue_session_requests(
                {req.req_id: [7, 8]},
                max_new_tokens=2,
                break_masks={req.req_id: [True, *([False] * 6)]},
            )
        engine.continue_session_requests(
            {req.req_id: [7, 8]},
            max_new_tokens=2,
            break_masks={req.req_id: [False] * 7},
        )
        self.assertIs(req.status, RequestStatus.RUNNING)
        self.assertEqual(req.prompt_token_ids, [1, 2, 3, 101, 999, 7, 8])
        self.assertEqual(req.output_token_ids, [])
        self.assertEqual(req.slots, retained_slots)
        self.assertIs(engine._caches[req.req_id], retained_cache)
        self.assertEqual(engine.metrics.session_reuse_hits, 0)
        self.assertEqual(engine.metrics.prefix_tokens_reused, 0)
        self.assertEqual(engine.metrics.continuation_input_tokens_computed, 0)

        engine.step()
        self.assertEqual(runner.prefill_chunk_calls[-1], ([999, 7, 8], 4))
        self.assertEqual(runner.caches_built, 1)
        self.assertEqual(engine.metrics.cache_builds, 1)
        self.assertEqual(engine.metrics.session_reuse_hits, 1)
        self.assertEqual(engine.metrics.prefix_tokens_reused, 4)
        self.assertEqual(engine.metrics.continuation_input_tokens_computed, 3)
        engine.step()
        self.assertIs(req.status, RequestStatus.PARKED)
        second_trace = engine.finished_traces[req.req_id].as_dict()
        self.assertEqual(second_trace["turn_index"], 1)
        self.assertEqual(second_trace["reused_prefix_tokens"], 4)
        self.assertEqual(second_trace["computed_input_tokens"], 3)
        self.assertEqual(engine.metrics.session_turns_completed, 2)

        closed = engine.close_sessions([req.req_id])
        self.assertEqual(closed, [req])
        self.assertIs(req.status, RequestStatus.FINISHED_CLOSED)
        self.assertEqual(engine.stats()["resident_sessions"], 0)
        self.assertEqual(engine.stats()["resident_state_slots"], 0)
        self.assertEqual(engine.stats()["token_pool"]["active_request_slots"], 0)

    def test_session_continuation_failure_releases_all_retained_state(self) -> None:
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
        )
        engine = GemmaNativeEngine(
            model=FakeModel(),
            config=cfg,
            num_slots=1,
            enable_token_pool_metadata=True,
            scheduler_config=SchedulerConfig(
                max_tokens_per_step=8,
                max_running_requests=1,
                max_tokens_per_request_per_step=8,
            ),
        )
        engine.runner = FakeFailingContinuationPrefillRunner()  # type: ignore[assignment]
        req = Request(
            prompt_token_ids=[1, 2, 3],
            max_new_tokens=2,
            req_id="session-error",
        )
        engine.add_session_request(req)
        engine.step()
        engine.step()
        self.assertIs(req.status, RequestStatus.PARKED)

        engine.continue_session_requests(
            {req.req_id: [7, 8]},
            max_new_tokens=2,
        )
        with self.assertRaisesRegex(RuntimeError, "synthetic continuation prefill failure"):
            engine.step()

        self.assertIs(req.status, RequestStatus.FINISHED_ERROR)
        self.assertNotIn(req.req_id, engine._session_req_ids)
        self.assertNotIn(req.req_id, engine._session_turn_indices)
        self.assertNotIn(req.req_id, engine._pending_session_reuse)
        self.assertNotIn(req.req_id, engine._caches)
        self.assertEqual(engine.stats()["resident_state_slots"], 0)
        self.assertEqual(engine.stats()["token_pool"]["active_request_slots"], 0)

    def test_token_pool_session_restores_exact_sliding_tail_for_continuation(self) -> None:
        import torch

        engine, runner = self._make_sliding_session_token_pool_engine(
            num_slots=1,
            prefill_microbatch_rows=1,
        )
        req = Request(
            prompt_token_ids=[1, 2, 3, 4, 5],
            max_new_tokens=3,
            req_id="session-tail",
        )
        engine.add_session_request(req)
        for _ in range(3):
            engine.step()
        self.assertIs(req.status, RequestStatus.PARKED)
        cache = engine._caches[req.req_id]
        self.assertIsNone(cache.layers[0].keys)
        tail_slots = engine._token_table.slots_for(req.req_id)[-3:]
        pooled_keys, pooled_values = engine._token_kv_pool.gather_kv(0, tail_slots)
        expected_keys = pooled_keys.permute(1, 0, 2).unsqueeze(0).contiguous()
        expected_values = pooled_values.permute(1, 0, 2).unsqueeze(0).contiguous()

        engine.continue_session_requests(
            {req.req_id: [7, 8]},
            max_new_tokens=2,
        )
        engine.step()

        self.assertEqual(len(runner.prefill_entry_tails), 1)
        cache_id, cumulative_length, restored_keys, restored_values = (
            runner.prefill_entry_tails[0]
        )
        self.assertEqual(cache_id, id(cache))
        self.assertEqual(cumulative_length, 7)
        self.assertTrue(torch.equal(restored_keys, expected_keys))
        self.assertTrue(torch.equal(restored_values, expected_values))
        self.assertEqual(engine.metrics.session_sliding_tail_restores, 1)
        self.assertEqual(engine.metrics.session_sliding_tail_tokens_restored, 3)
        self.assertIsNone(cache.layers[0].keys)
        self.assertTrue(cache.layers[0]._dense_storage_released)

        engine.step()
        self.assertIs(req.status, RequestStatus.PARKED)
        engine.close_sessions([req.req_id])
        self.assertEqual(engine.stats()["token_pool"]["active_request_slots"], 0)

    def test_batched_token_pool_sessions_restore_each_continuation_tail(self) -> None:
        import torch

        engine, runner = self._make_sliding_session_token_pool_engine(
            num_slots=2,
            prefill_microbatch_rows=2,
        )
        reqs = [
            Request(
                prompt_token_ids=[1, 2, 3, 4, 5],
                max_new_tokens=3,
                req_id=f"session-tail-{row}",
            )
            for row in range(2)
        ]
        for req in reqs:
            engine.add_session_request(req)
        for _ in range(3):
            engine.step()
        self.assertTrue(all(req.status is RequestStatus.PARKED for req in reqs))

        expected_by_cache = {}
        for req in reqs:
            cache = engine._caches[req.req_id]
            tail_slots = engine._token_table.slots_for(req.req_id)[-3:]
            pooled_keys, pooled_values = engine._token_kv_pool.gather_kv(0, tail_slots)
            expected_by_cache[id(cache)] = (
                pooled_keys.permute(1, 0, 2).unsqueeze(0).contiguous(),
                pooled_values.permute(1, 0, 2).unsqueeze(0).contiguous(),
            )

        engine.continue_session_requests(
            {req.req_id: [7, 8] for req in reqs},
            max_new_tokens=2,
        )
        engine.step()

        self.assertEqual(len(runner.prefill_entry_tails), 2)
        for cache_id, cumulative_length, restored_keys, restored_values in (
            runner.prefill_entry_tails
        ):
            expected_keys, expected_values = expected_by_cache[cache_id]
            self.assertEqual(cumulative_length, 7)
            self.assertTrue(torch.equal(restored_keys, expected_keys))
            self.assertTrue(torch.equal(restored_values, expected_values))
        self.assertEqual(engine.metrics.session_sliding_tail_restores, 2)
        self.assertEqual(engine.metrics.session_sliding_tail_tokens_restored, 6)

        engine.step()
        self.assertTrue(all(req.status is RequestStatus.PARKED for req in reqs))
        engine.close_sessions([req.req_id for req in reqs])
        self.assertEqual(engine.stats()["token_pool"]["active_request_slots"], 0)

    def test_decode_batch_uses_one_runner_call_for_compatible_rows(self) -> None:
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=2,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention", "sliding_attention"),
        )
        engine = GemmaNativeEngine(
            model=FakeModel(),
            config=cfg,
            num_slots=2,
            enable_token_pool_metadata=True,
            scheduler_config=SchedulerConfig(
                max_tokens_per_step=16,
                max_running_requests=2,
                max_tokens_per_request_per_step=8,
            ),
        )
        runner = FakeBatchRunner()
        engine.runner = runner  # type: ignore[assignment]

        reqs = [
            Request(prompt_token_ids=[1, 2, 3], max_new_tokens=3, req_id="a"),
            Request(prompt_token_ids=[4, 5, 6], max_new_tokens=3, req_id="b"),
        ]
        for req in reqs:
            engine.add_request(req)

        engine.step()
        self.assertEqual([req.output_token_ids for req in reqs], [[101], [102]])

        engine.step()
        self.assertEqual(len(runner.decode_batch_calls), 1)
        self.assertEqual(runner.decode_step_calls, 0)
        self.assertEqual(runner.decode_batch_calls[0], ([101, 102], [3, 3]))
        self.assertEqual([req.output_token_ids for req in reqs], [[101, 200], [102, 201]])
        self.assertEqual(engine.metrics.decode_model_calls, 1)
        self.assertEqual(engine.metrics.batched_decode_model_calls, 1)
        self.assertEqual(engine.metrics.fallback_decode_model_calls, 0)
        self.assertEqual(engine.metrics.exact_decode_batch_rows, 2)
        self.assertEqual(engine.metrics.max_decode_model_batch_rows, 2)

    def test_decode_batch_builds_token_pool_metadata(self) -> None:
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=2,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention", "sliding_attention"),
            sliding_window=2,
        )
        engine = GemmaNativeEngine(
            model=FakeModel(),
            config=cfg,
            num_slots=2,
            enable_token_pool_metadata=True,
            scheduler_config=SchedulerConfig(
                max_tokens_per_step=16,
                max_running_requests=2,
                max_tokens_per_request_per_step=8,
            ),
        )
        runner = FakeBatchRunner()
        engine.runner = runner  # type: ignore[assignment]
        reqs = [
            Request(prompt_token_ids=[1, 2, 3], max_new_tokens=3, req_id="a"),
            Request(prompt_token_ids=[4, 5, 6], max_new_tokens=3, req_id="b"),
        ]
        for req in reqs:
            engine.add_request(req)

        engine.step()
        self.assertEqual(engine.stats()["token_pool"]["allocated_token_slots"], 6)

        engine.step()

        metadata = engine.last_token_pool_decode_metadata
        self.assertIsNotNone(metadata)
        full = metadata["full_attention"]  # type: ignore[index]
        sliding = metadata["sliding_attention"]  # type: ignore[index]
        self.assertEqual(full.req_pool_indices.tolist(), [0, 1])
        self.assertEqual(full.logical_seq_lens.tolist(), [4, 4])
        self.assertEqual(full.seq_lens.tolist(), [4, 4])
        self.assertEqual(full.out_cache_loc.tolist(), [6, 7])
        self.assertEqual(full.kv_indptr.tolist(), [0, 4, 8])
        self.assertEqual(full.kv_indices.tolist(), [0, 1, 2, 6, 3, 4, 5, 7])
        self.assertEqual(sliding.logical_seq_lens.tolist(), [4, 4])
        self.assertEqual(sliding.seq_lens.tolist(), [2, 2])
        self.assertEqual(sliding.kv_indptr.tolist(), [0, 2, 4])
        self.assertEqual(sliding.kv_indices.tolist(), [2, 6, 5, 7])
        self.assertEqual(len(runner.decode_batch_token_pool_contexts), 1)
        self.assertIsNone(runner.decode_batch_token_pool_contexts[0])
        self.assertEqual(engine.metrics.token_pool_decode_metadata_batches, 1)
        self.assertEqual(engine.metrics.token_pool_decode_metadata_rows, 2)
        self.assertEqual(engine.stats()["token_pool"]["allocated_token_slots"], 8)

    def test_token_pool_attention_builds_and_backfills_sliding_kv_pool(self) -> None:
        import torch
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config
        from wkvm.runner.gemma_token_pool import (
            TokenPoolDecodeBackendState,
            TokenPoolPreparedDecodeBatch,
        )

        class FakeNativeTokenPoolModel:
            wkvm_no_hf_transformer_forward = True

            def __init__(self) -> None:
                self._param = torch.empty((), dtype=torch.float32)
                self.config = SimpleNamespace(num_attention_heads=2)
                attn = SimpleNamespace(
                    layer_type="sliding_attention",
                    is_kv_shared_layer=False,
                    num_key_value_groups=2,
                    head_dim=4,
                )
                self.text_prefix = SimpleNamespace(
                    layers=[SimpleNamespace(layer_idx=0, attn_meta=attn)]
                )

            def parameters(self):
                return iter([self._param])

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=4,
        )
        engine = GemmaNativeEngine(
            model=FakeNativeTokenPoolModel(),
            config=cfg,
            num_slots=1,
            enable_token_pool_attention=True,
            token_pool_max_context_len=4,
            token_pool_capacity=32,
        )
        self.assertIsInstance(
            engine._token_pool_decode_backend,
            TokenPoolDecodeBackendState,
        )
        self.assertIs(
            engine._token_pool_decode_backend.block_tables,
            engine._token_pool_block_tables,
        )
        req = Request(prompt_token_ids=[1, 2, 3], max_new_tokens=2, req_id="tp")
        keys = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
        values = keys + 100
        cache = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    is_sliding=True,
                    keys=keys,
                    values=values,
                )
            ]
        )

        engine._token_pool_commit_prefill_tokens(req, 3, cache=cache)
        req.num_computed_tokens = 3
        req.output_token_ids.append(9)
        reservations = engine._token_pool_prepare_decode_batch([req])
        self.assertIsInstance(reservations, TokenPoolPreparedDecodeBatch)

        def fail_context_rewrap(*args, **kwargs):
            raise AssertionError("prepared decode batch should build its own context")

        engine._token_pool_decode_backend.build_decode_context_for_batch = (  # type: ignore[method-assign,union-attr]
            fail_context_rewrap
        )
        context = engine._token_pool_decode_context(reservations)

        self.assertIsNotNone(context)
        self.assertIs(context.kv_pool, engine._token_kv_pool)
        self.assertIsNotNone(engine._token_kv_pool)
        gathered_k, gathered_v = engine._token_kv_pool.gather_kv(0, [0, 1, 2])
        self.assertTrue(torch.equal(gathered_k, keys[0].permute(1, 0, 2)))
        self.assertTrue(torch.equal(gathered_v, values[0].permute(1, 0, 2)))
        paged = context.paged_metadata_for_layer(0, "sliding_attention")
        self.assertIsNotNone(paged)
        self.assertEqual(paged.seq_lens.tolist(), [4])
        self.assertEqual(paged.selected_start_positions.tolist(), [0])
        self.assertEqual(paged.block_tables.tolist(), [[0, -1]])
        self.assertEqual(paged.block_table_lens.tolist(), [1])
        self.assertEqual(paged.out_cache_loc.tolist(), [3])
        stats = engine.stats()["token_pool"]
        self.assertTrue(stats["attention_enabled"])
        self.assertEqual(stats["kv_pool_layers"], 1)
        self.assertEqual(stats["page_table_tensor_shape"], (1, 1))
        self.assertEqual(stats["block_table_bytes"], 4)
        self.assertIsNotNone(engine._token_pool_block_tables)
        page_table_tensor = engine._token_pool_page_table_tensor
        self.assertIsNotNone(page_table_tensor)
        self.assertIs(page_table_tensor, engine._token_pool_block_tables.tensor)
        req_slot = engine._token_pool_req_slots[req.req_id]
        self.assertEqual(page_table_tensor[req_slot, :1].tolist(), [0])
        engine._token_pool_release_request(req.req_id)
        self.assertEqual(page_table_tensor[req_slot, :1].tolist(), [-1])

    def test_one_shot_prefill_uses_authoritative_sliding_pool_path(self) -> None:
        import torch
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        class FakeNativeTokenPoolModel:
            wkvm_no_hf_transformer_forward = True

            def __init__(self) -> None:
                self._param = torch.empty((), dtype=torch.float32)
                self.config = SimpleNamespace(num_attention_heads=2)
                attn = SimpleNamespace(
                    layer_type="sliding_attention",
                    is_kv_shared_layer=False,
                    num_key_value_groups=2,
                    head_dim=4,
                )
                self.text_prefix = SimpleNamespace(
                    layers=[SimpleNamespace(layer_idx=0, attn_meta=attn)]
                )

            def parameters(self):
                return iter([self._param])

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            num_kv_heads=1,
            head_dim=4,
            sliding_window=4,
        )
        model = FakeNativeTokenPoolModel()
        engine = GemmaNativeEngine(
            model=model,
            config=cfg,
            num_slots=1,
            enable_token_pool_attention=True,
            token_pool_max_context_len=8,
            token_pool_capacity=32,
        )
        hf_config = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=4,
        )
        runner = FakeAuthoritativePrefillRunner(model, hf_config, cfg)
        engine.runner = runner
        backend = engine._token_pool_decode_backend
        self.assertIsNotNone(backend)

        def fail_backfill(*args, **kwargs):
            raise AssertionError("one-shot authoritative prefill must not backfill")

        backend.backfill_prefill_tokens = fail_backfill  # type: ignore[method-assign,union-attr]
        req = Request(
            prompt_token_ids=[1, 2, 3, 4, 5],
            max_new_tokens=2,
            req_id="one-shot",
        )
        engine.add_request(req)

        engine.step()

        cache = runner.last_cache
        self.assertIsNotNone(cache)
        self.assertEqual(engine.metrics.token_pool_authoritative_prefill_requests, 1)
        self.assertEqual(engine.metrics.token_pool_authoritative_prefill_tokens, 5)
        self.assertEqual(engine.metrics.token_pool_authoritative_prefill_layer_writes, 1)
        self.assertEqual(
            engine._token_table.slots_for("one-shot").tolist(),
            [-1, -1, 2, 3, 4],
        )
        pooled_k, pooled_v = engine._token_kv_pool.gather_kv(0, [2, 3, 4])
        self.assertTrue(
            torch.equal(
                pooled_k,
                runner.last_keys[0, :, -3:, :].permute(1, 0, 2),
            )
        )
        self.assertTrue(
            torch.equal(
                pooled_v,
                runner.last_values[0, :, -3:, :].permute(1, 0, 2),
            )
        )
        self.assertIsNone(cache.layers[0].keys)
        self.assertIsNone(cache.layers[0].values)
        self.assertEqual(cache._shared_kv_by_layer, {})
        self.assertEqual(cache._shared_kv_by_type, {})
        self.assertEqual(cache.state_bytes(), 0)

    def test_chunked_prefill_keeps_authoritative_metric_zero(self) -> None:
        import torch
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        class FakeNativeTokenPoolModel:
            wkvm_no_hf_transformer_forward = True

            def __init__(self) -> None:
                self._param = torch.empty((), dtype=torch.float32)
                self.config = SimpleNamespace(num_attention_heads=2)
                attn = SimpleNamespace(
                    layer_type="sliding_attention",
                    is_kv_shared_layer=False,
                    num_key_value_groups=2,
                    head_dim=4,
                )
                self.text_prefix = SimpleNamespace(
                    layers=[SimpleNamespace(layer_idx=0, attn_meta=attn)]
                )

            def parameters(self):
                return iter([self._param])

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            num_kv_heads=1,
            head_dim=4,
            sliding_window=4,
        )
        model = FakeNativeTokenPoolModel()
        engine = GemmaNativeEngine(
            model=model,
            config=cfg,
            num_slots=1,
            prefill_chunk=3,
            enable_token_pool_attention=True,
            token_pool_max_context_len=8,
            token_pool_capacity=32,
        )
        hf_config = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=4,
        )
        runner = FakeAuthoritativePrefillRunner(model, hf_config, cfg)
        engine.runner = runner
        backend = engine._token_pool_decode_backend
        self.assertIsNotNone(backend)
        original_backfill = backend.backfill_prefill_tokens
        backfill_calls = 0

        def count_backfill(*args, **kwargs):
            nonlocal backfill_calls
            backfill_calls += 1
            return original_backfill(*args, **kwargs)

        backend.backfill_prefill_tokens = count_backfill  # type: ignore[method-assign,union-attr]
        req = Request(
            prompt_token_ids=[1, 2, 3, 4, 5],
            max_new_tokens=2,
            req_id="chunked",
        )
        engine.add_request(req)

        engine.step()
        self.assertEqual(engine.metrics.token_pool_authoritative_prefill_requests, 0)
        self.assertIsNotNone(runner.last_cache.layers[0].keys)
        engine.step()

        self.assertEqual(engine.metrics.token_pool_authoritative_prefill_requests, 0)
        self.assertEqual(engine.metrics.token_pool_authoritative_prefill_tokens, 0)
        self.assertEqual(engine.metrics.token_pool_authoritative_prefill_layer_writes, 0)
        self.assertEqual(backfill_calls, 2)

    def test_token_pool_attention_pages_mid_block_sliding_tail(self) -> None:
        import torch
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        class FakeNativeTokenPoolModel:
            wkvm_no_hf_transformer_forward = True

            def __init__(self) -> None:
                self._param = torch.empty((), dtype=torch.float32)
                self.config = SimpleNamespace(num_attention_heads=2)
                attn = SimpleNamespace(
                    layer_type="sliding_attention",
                    is_kv_shared_layer=False,
                    num_key_value_groups=2,
                    head_dim=4,
                )
                self.text_prefix = SimpleNamespace(
                    layers=[SimpleNamespace(layer_idx=0, attn_meta=attn)]
                )

            def parameters(self):
                return iter([self._param])

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=17,
        )
        engine = GemmaNativeEngine(
            model=FakeNativeTokenPoolModel(),
            config=cfg,
            num_slots=1,
            enable_token_pool_attention=True,
            token_pool_max_context_len=17,
            token_pool_capacity=64,
        )
        req = Request(prompt_token_ids=list(range(17)), max_new_tokens=2, req_id="tp")
        keys = torch.arange(17 * 4, dtype=torch.float32).reshape(1, 1, 17, 4)
        cache = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    is_sliding=True,
                    keys=keys,
                    values=keys + 100,
                )
            ]
        )

        engine._token_pool_commit_prefill_tokens(req, 17, cache=cache)
        req_slot = engine._token_pool_req_slots[req.req_id]
        engine._token_pool_clear_prefix(req.req_id, req_slot, 1)
        req.num_computed_tokens = 17
        req.output_token_ids.append(9)

        reservations = engine._token_pool_prepare_decode_batch([req])
        context = engine._token_pool_decode_context(reservations)

        self.assertIsNotNone(context)
        sliding = context.metadata_for_layer(0, "sliding_attention")
        self.assertIsNotNone(sliding)
        self.assertEqual(sliding.out_cache_loc.tolist(), [17])
        paged = context.paged_metadata_for_layer(0, "sliding_attention")
        self.assertIsNotNone(paged)
        self.assertEqual(paged.seq_lens.tolist(), [17])
        self.assertEqual(paged.selected_start_positions.tolist(), [1])
        self.assertEqual(paged.block_tables.tolist(), [[0, 1]])
        self.assertEqual(paged.block_table_lens.tolist(), [2])
        self.assertEqual(paged.out_cache_loc.tolist(), [17])
        page_table_tensor = engine._token_pool_page_table_tensor
        self.assertIsNotNone(page_table_tensor)
        self.assertEqual(page_table_tensor[req_slot, :2].tolist(), [0, 1])

    def test_token_pool_attention_accepts_custom_paged_block_size(self) -> None:
        import torch
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        class FakeNativeTokenPoolModel:
            wkvm_no_hf_transformer_forward = True

            def __init__(self) -> None:
                self._param = torch.empty((), dtype=torch.float32)
                self.config = SimpleNamespace(num_attention_heads=2)
                attn = SimpleNamespace(
                    layer_type="sliding_attention",
                    is_kv_shared_layer=False,
                    num_key_value_groups=2,
                    head_dim=4,
                )
                self.text_prefix = SimpleNamespace(
                    layers=[SimpleNamespace(layer_idx=0, attn_meta=attn)]
                )

            def parameters(self):
                return iter([self._param])

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=9,
        )
        engine = GemmaNativeEngine(
            model=FakeNativeTokenPoolModel(),
            config=cfg,
            num_slots=1,
            enable_token_pool_attention=True,
            token_pool_max_context_len=16,
            token_pool_capacity=32,
            token_pool_paged_block_size=8,
        )
        req = Request(prompt_token_ids=list(range(9)), max_new_tokens=2, req_id="tp8")
        keys = torch.arange(9 * 4, dtype=torch.float32).reshape(1, 1, 9, 4)
        cache = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    is_sliding=True,
                    keys=keys,
                    values=keys + 100,
                )
            ]
        )

        engine._token_pool_commit_prefill_tokens(req, 9, cache=cache)
        req_slot = engine._token_pool_req_slots[req.req_id]
        engine._token_pool_clear_prefix(req.req_id, req_slot, 1)
        req.num_computed_tokens = 9
        req.output_token_ids.append(9)

        reservations = engine._token_pool_prepare_decode_batch([req])
        context = engine._token_pool_decode_context(reservations)

        self.assertIsNotNone(context)
        paged = context.paged_metadata_for_layer(0, "sliding_attention")
        self.assertIsNotNone(paged)
        self.assertEqual(paged.block_size, 8)
        self.assertEqual(paged.seq_lens.tolist(), [9])
        self.assertEqual(paged.selected_start_positions.tolist(), [1])
        self.assertEqual(paged.block_tables.tolist(), [[0, 1]])
        self.assertEqual(paged.block_table_lens.tolist(), [2])
        self.assertEqual(paged.out_cache_loc.tolist(), [9])
        self.assertEqual(engine.stats()["token_pool"]["paged_block_size"], 8)

    def test_token_pool_attention_page32_metadata_crosses_block_boundary(self) -> None:
        import torch
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        class FakeNativeTokenPoolModel:
            wkvm_no_hf_transformer_forward = True

            def __init__(self) -> None:
                self._param = torch.empty((), dtype=torch.float32)
                self.config = SimpleNamespace(num_attention_heads=2)
                attn = SimpleNamespace(
                    layer_type="sliding_attention",
                    is_kv_shared_layer=False,
                    num_key_value_groups=2,
                    head_dim=4,
                )
                self.text_prefix = SimpleNamespace(
                    layers=[SimpleNamespace(layer_idx=0, attn_meta=attn)]
                )

            def parameters(self):
                return iter([self._param])

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=33,
        )
        engine = GemmaNativeEngine(
            model=FakeNativeTokenPoolModel(),
            config=cfg,
            num_slots=1,
            enable_token_pool_attention=True,
            token_pool_max_context_len=64,
            token_pool_capacity=96,
            token_pool_paged_block_size=32,
        )
        req = Request(prompt_token_ids=list(range(33)), max_new_tokens=2, req_id="tp32")
        keys = torch.arange(33 * 4, dtype=torch.float32).reshape(1, 1, 33, 4)
        cache = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    is_sliding=True,
                    keys=keys,
                    values=keys + 100,
                )
            ]
        )

        engine._token_pool_commit_prefill_tokens(req, 33, cache=cache)
        req_slot = engine._token_pool_req_slots[req.req_id]
        engine._token_pool_clear_prefix(req.req_id, req_slot, 1)
        req.num_computed_tokens = 33
        req.output_token_ids.append(9)

        reservations = engine._token_pool_prepare_decode_batch([req])
        context = engine._token_pool_decode_context(reservations)

        self.assertIsNotNone(context)
        paged = context.paged_metadata_for_layer(0, "sliding_attention")
        self.assertIsNotNone(paged)
        self.assertEqual(paged.block_size, 32)
        self.assertEqual(paged.seq_lens.tolist(), [33])
        self.assertEqual(paged.logical_seq_lens.tolist(), [34])
        self.assertEqual(paged.selected_start_positions.tolist(), [1])
        self.assertEqual(paged.block_tables.tolist(), [[0, 1]])
        self.assertEqual(paged.block_table_lens.tolist(), [2])
        self.assertEqual(paged.out_cache_loc.tolist(), [33])

    def test_sliding_decode_metadata_padding_reserves_future_steps(self) -> None:
        import torch
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        class FakeNativeTokenPoolModel:
            wkvm_no_hf_transformer_forward = True

            def __init__(self) -> None:
                self._param = torch.empty((), dtype=torch.float32)
                self.config = SimpleNamespace(num_attention_heads=2)
                attn = SimpleNamespace(
                    layer_type="sliding_attention",
                    is_kv_shared_layer=False,
                    num_key_value_groups=2,
                    head_dim=4,
                )
                self.text_prefix = SimpleNamespace(
                    layers=[SimpleNamespace(layer_idx=0, attn_meta=attn)]
                )

            def parameters(self):
                return iter([self._param])

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=4,
        )
        engine = GemmaNativeEngine(
            model=FakeNativeTokenPoolModel(),
            config=cfg,
            num_slots=1,
            enable_token_pool_attention=True,
            token_pool_max_context_len=4,
            token_pool_capacity=32,
        )
        req = Request(prompt_token_ids=[1], max_new_tokens=4, req_id="slide")
        keys = torch.arange(4, dtype=torch.float32).reshape(1, 1, 1, 4)
        cache = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    is_sliding=True,
                    keys=keys,
                    values=keys + 100,
                )
            ]
        )
        engine._token_pool_commit_prefill_tokens(req, 1, cache=cache)
        req.num_computed_tokens = 1
        req.output_token_ids.append(9)

        reservations = engine._token_pool_prepare_decode_batch(
            [req],
            sliding_attention_kv_indices_padding_steps=2,
        )
        context = engine._token_pool_decode_context(reservations)

        self.assertIsNotNone(context)
        sliding = context.metadata_by_layer_type["sliding_attention"]  # type: ignore[union-attr]
        valid_total = int(sliding.kv_indptr[-1].item())
        self.assertEqual(sliding.seq_lens.tolist(), [2])
        self.assertEqual(valid_total, 2)
        self.assertEqual(int(sliding.kv_indices.numel()), 4)
        self.assertEqual(
            sliding.kv_indices[valid_total:].tolist(),
            [int(sliding.kv_indices[valid_total - 1].item())] * 2,
        )

        engine._token_pool_discard_decode_reservations(reservations)
        engine._token_pool_release_request(req.req_id)
        self.assertEqual(engine.stats()["token_pool"]["allocated_token_slots"], 0)

    def test_token_kv_layer_specs_include_full_and_shared_aliases(self) -> None:
        import torch
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        class FakeNativeTokenPoolModel:
            wkvm_no_hf_transformer_forward = True

            def __init__(self) -> None:
                self._param = torch.empty((), dtype=torch.float32)
                self.config = SimpleNamespace(num_attention_heads=4)
                metas = [
                    SimpleNamespace(
                        layer_type="sliding_attention",
                        is_kv_shared_layer=False,
                        kv_shared_layer_index=None,
                        num_key_value_groups=2,
                        head_dim=4,
                    ),
                    SimpleNamespace(
                        layer_type="full_attention",
                        is_kv_shared_layer=False,
                        kv_shared_layer_index=None,
                        num_key_value_groups=1,
                        head_dim=8,
                    ),
                    SimpleNamespace(
                        layer_type="sliding_attention",
                        is_kv_shared_layer=True,
                        kv_shared_layer_index=0,
                        num_key_value_groups=2,
                        head_dim=4,
                    ),
                    SimpleNamespace(
                        layer_type="full_attention",
                        is_kv_shared_layer=True,
                        kv_shared_layer_index=1,
                        num_key_value_groups=1,
                        head_dim=8,
                    ),
                ]
                self.text_prefix = SimpleNamespace(
                    layers=[
                        SimpleNamespace(layer_idx=idx, attn_meta=meta)
                        for idx, meta in enumerate(metas)
                    ]
                )

            def parameters(self):
                return iter([self._param])

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=4,
            num_kv_shared_layers=2,
            layer_types=(
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ),
            sliding_window=4,
        )
        engine = GemmaNativeEngine(
            model=FakeNativeTokenPoolModel(),
            config=cfg,
            num_slots=1,
            enable_token_pool_attention=True,
            token_pool_max_context_len=4,
            token_pool_capacity=4,
        )

        pool = engine._token_kv_pool
        self.assertIsNotNone(pool)
        specs = pool.layer_specs
        self.assertEqual(set(specs), {0, 1, 2, 3})
        self.assertEqual(pool.target_layer(0), 0)
        self.assertEqual(pool.target_layer(1), 1)
        self.assertEqual(pool.target_layer(2), 0)
        self.assertEqual(pool.target_layer(3), 1)
        self.assertEqual(specs[2].kv_share_target_layer, 0)
        self.assertEqual(specs[3].kv_share_target_layer, 1)
        self.assertEqual(specs[0].num_kv_heads, 2)
        self.assertEqual(specs[1].num_kv_heads, 4)
        self.assertEqual(specs[1].head_dim, 8)
        self.assertEqual(pool.allocated_layer_count, 0)
        self.assertEqual(pool.state_bytes(), 0)
        self.assertEqual(engine.stats()["token_pool"]["kv_pool_layers"], 4)

    def test_token_pool_attention_builds_full_layer_rows_by_layer_id(self) -> None:
        import torch
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config
        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        class FakeNativeTokenPoolModel:
            wkvm_no_hf_transformer_forward = True

            def __init__(self) -> None:
                self._param = torch.empty((), dtype=torch.float32)
                self.config = SimpleNamespace(
                    num_attention_heads=2,
                    layer_types=("sliding_attention", "full_attention"),
                )
                metas = [
                    SimpleNamespace(
                        layer_type="sliding_attention",
                        is_kv_shared_layer=False,
                        num_key_value_groups=2,
                        head_dim=2,
                    ),
                    SimpleNamespace(
                        layer_type="full_attention",
                        is_kv_shared_layer=False,
                        num_key_value_groups=2,
                        head_dim=2,
                    ),
                ]
                self.text_prefix = SimpleNamespace(
                    layers=[
                        SimpleNamespace(layer_idx=idx, attn_meta=meta)
                        for idx, meta in enumerate(metas)
                    ]
                )

            def parameters(self):
                return iter([self._param])

        hf_cfg = SimpleNamespace(
            num_hidden_layers=2,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention", "full_attention"),
            sliding_window=8,
        )
        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=2,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention", "full_attention"),
            sink_tokens=1,
            ring_tokens=1,
            pending_tokens=2,
            routed_slots=2,
            reps_per_slot=1,
            span_budget_tokens=4,
            max_span_tokens=3,
            sliding_window=8,
        )
        engine = GemmaNativeEngine(
            model=FakeNativeTokenPoolModel(),
            config=cfg,
            num_slots=1,
            enable_token_pool_attention=True,
            token_pool_max_context_len=8,
            token_pool_capacity=32,
        )
        req = Request(prompt_token_ids=[1, 2, 3, 4, 5], max_new_tokens=2, req_id="full")
        cache = NativeGemmaRoutedCache(hf_cfg, cfg)
        sliding_keys = torch.arange(10, dtype=torch.float32).reshape(1, 1, 5, 2)
        full_keys = torch.arange(20, 30, dtype=torch.float32).reshape(1, 1, 5, 2)
        cache.update(sliding_keys, sliding_keys + 100, layer_idx=0)
        cache.update(full_keys, full_keys + 100, layer_idx=1)
        engine._caches[req.req_id] = cache
        expected_full_keys = cache.layers[1].keys.clone()
        expected_full_values = cache.layers[1].values.clone()

        engine._token_pool_commit_prefill_tokens(
            req,
            req.num_prompt_tokens,
            cache=cache,
            final_prefill=True,
        )
        req.num_computed_tokens = req.num_prompt_tokens
        req.output_token_ids.append(77)

        reservations = engine._token_pool_prepare_decode_batch([req])
        context = engine._token_pool_decode_context(reservations)

        self.assertIsNotNone(context)
        self.assertEqual(reservations[0].token_slot_tensor.tolist(), [5])
        self.assertEqual(reservations[0].token_slot_tensor.dtype, torch.int32)
        full_metadata = context.metadata_by_layer_id[1]  # type: ignore[index]
        sliding_metadata = context.metadata_by_layer_id[0]  # type: ignore[index]
        full_metadata_ptrs = {
            name: int(getattr(full_metadata, name).data_ptr())
            for name in (
                "req_pool_indices",
                "seq_lens",
                "logical_seq_lens",
                "out_cache_loc",
                "kv_indptr",
                "kv_indices",
                "out_cache_loc_long",
            )
        }
        self.assertIs(full_metadata, context.metadata_by_layer_type["full_attention"])  # type: ignore[union-attr]
        self.assertIs(sliding_metadata, context.metadata_by_layer_type["sliding_attention"])  # type: ignore[union-attr]
        full_layer = cache.layers[1]
        materialized_width = full_layer.materialized_tokens()
        self.assertGreater(materialized_width, 0)
        self.assertLessEqual(materialized_width, cfg.routed_materialized_tokens)
        self.assertEqual(full_metadata.seq_lens.tolist(), [materialized_width + 1])
        self.assertEqual(full_metadata.logical_seq_lens.tolist(), [req.num_prompt_tokens + 1])
        self.assertEqual(full_metadata.out_cache_loc.tolist(), [5])
        self.assertEqual(full_metadata.kv_indices.tolist()[-1], 5)
        full_metadata_kv_indices = full_metadata.kv_indices.tolist()
        full_metadata_out_cache_loc = full_metadata.out_cache_loc.tolist()
        self.assertEqual(
            engine.last_token_pool_decode_covered_layer_types,
            frozenset({"sliding_attention", "full_attention"}),
        )
        self.assertEqual(
            engine.metrics.token_pool_decode_covered_layer_type_batches,
            {"sliding_attention": 1, "full_attention": 1},
        )
        self.assertEqual(
            engine.metrics.token_pool_decode_covered_layer_type_rows,
            {"sliding_attention": 1, "full_attention": 1},
        )
        self.assertEqual(engine._token_table.slots_for("full").tolist(), [0, 1, 2, 3, 4, 5])
        self.assertNotEqual(
            full_metadata.kv_indices.tolist()[:materialized_width],
            engine._token_table.slots_for("full")[:materialized_width].tolist(),
        )
        gathered_k, gathered_v = engine._token_kv_pool.gather_kv(  # type: ignore[union-attr]
            1,
            full_metadata.kv_indices[:-1],
        )
        self.assertTrue(torch.equal(gathered_k, expected_full_keys[0].permute(1, 0, 2)))
        self.assertTrue(torch.equal(gathered_v, expected_full_values[0].permute(1, 0, 2)))
        self.assertIsNotNone(full_layer.keys)
        self.assertIsNotNone(full_layer.values)
        self.assertFalse(full_layer._dense_storage_released)

        engine._token_pool_discard_decode_reservations(reservations)
        self.assertEqual(engine._token_pool_full_attention_slots, {})
        self.assertEqual(engine.stats()["token_pool"]["allocated_token_slots"], 16)

        repeat_reservations = engine._token_pool_prepare_decode_batch([req])
        repeat_context = engine._token_pool_decode_context(repeat_reservations)
        self.assertIsNotNone(repeat_context)
        repeat_full = repeat_context.metadata_by_layer_type["full_attention"]  # type: ignore[union-attr]
        self.assertEqual(repeat_full.kv_indices.tolist(), full_metadata_kv_indices)
        self.assertEqual(repeat_full.out_cache_loc.tolist(), full_metadata_out_cache_loc)
        for name, ptr in full_metadata_ptrs.items():
            self.assertEqual(int(getattr(repeat_full, name).data_ptr()), ptr)
        engine._token_pool_discard_decode_reservations(repeat_reservations)
        self.assertEqual(engine._token_pool_full_attention_slots, {})
        self.assertEqual(engine.stats()["token_pool"]["allocated_token_slots"], 16)

        padded_reservations = engine._token_pool_prepare_decode_batch(
            [req],
            full_attention_kv_indices_padding_steps=2,
        )
        padded_context = engine._token_pool_decode_context(padded_reservations)
        self.assertIsNotNone(padded_context)
        padded_full = padded_context.metadata_by_layer_type["full_attention"]  # type: ignore[union-attr]
        valid_total = int(padded_full.kv_indptr[-1].item())
        self.assertEqual(int(padded_full.kv_indices.numel()), valid_total + 2)
        self.assertEqual(
            padded_full.kv_indices[valid_total:].tolist(),
            [int(padded_full.kv_indices[valid_total - 1].item())] * 2,
        )
        engine._token_pool_discard_decode_reservations(padded_reservations)
        self.assertEqual(engine.stats()["token_pool"]["allocated_token_slots"], 16)

    def test_nonpersistent_full_attention_row_releases_dense_readout(self) -> None:
        engine, req, cache = self._make_full_attention_token_pool_fixture(
            req_id="nonpersistent-release"
        )
        full_layer = cache.layers[1]
        self.assertIsNotNone(full_layer.keys)

        reservations = engine._token_pool_prepare_decode_batch([req])
        context = engine._token_pool_decode_context(reservations)

        self.assertIsNotNone(context)
        self.assertIn("full_attention", context.covered_layer_types)
        self.assertIsNone(full_layer.keys)
        self.assertIsNone(full_layer.values)
        self.assertTrue(full_layer._dense_storage_released)

        engine._token_pool_discard_decode_reservations(reservations)
        repeated = engine._token_pool_prepare_decode_batch([req])
        repeated_context = engine._token_pool_decode_context(repeated)
        self.assertIsNotNone(repeated_context)
        self.assertIn("full_attention", repeated_context.covered_layer_types)
        self.assertIsNone(full_layer.keys)
        self.assertIsNone(full_layer.values)
        engine._token_pool_discard_decode_reservations(repeated)
        engine._token_pool_release_request(req.req_id)

    def test_persistent_full_attention_rows_reuse_materialized_slots(self) -> None:
        import torch

        engine, req, cache = self._make_full_attention_token_pool_fixture(
            req_id="persist"
        )
        full_layer = cache.layers[1]
        materialized_width = full_layer.materialized_tokens()

        reservations = engine._token_pool_prepare_decode_batch(
            [req],
            full_attention_kv_indices_padding_steps=1,
            persistent_full_attention_rows=True,
        )
        context = engine._token_pool_decode_context(reservations)
        self.assertIsNotNone(context)
        full_metadata = context.metadata_by_layer_type["full_attention"]  # type: ignore[union-attr]
        first_metadata_ptrs = {
            name: int(getattr(full_metadata, name).data_ptr())
            for name in (
                "req_pool_indices",
                "seq_lens",
                "logical_seq_lens",
                "out_cache_loc",
                "kv_indptr",
                "kv_indices",
                "out_cache_loc_long",
            )
        }
        self.assertEqual(full_metadata.seq_lens.tolist(), [materialized_width + 1])
        first_full_slot = reservations[0].full_attention_token_slot
        self.assertIsNotNone(first_full_slot)
        self.assertEqual(first_full_slot, reservations[0].token_slot)
        self.assertEqual(full_metadata.out_cache_loc.tolist(), [first_full_slot])
        self.assertEqual(engine.metrics.token_pool_full_attention_row_rebuilds, 1)
        self.assertEqual(engine.metrics.token_pool_full_attention_row_reuses, 0)
        self.assertEqual(engine.metrics.token_pool_full_attention_row_appends, 0)
        self.assertIsNone(full_layer.keys)
        self.assertIsNone(full_layer.values)

        decode_key = torch.tensor([[[301.0, 302.0]]])
        engine._token_kv_pool.set_kv(  # type: ignore[union-attr]
            1,
            [first_full_slot],
            decode_key,
            decode_key + 100,
        )
        engine._token_pool_commit_decode_reservations(reservations)
        self.assertIsNone(full_layer.keys)
        self.assertIsNone(full_layer.values)

        row = engine._token_pool_full_attention_rows["persist"]
        self.assertEqual(row.owned_slots, [])
        self.assertEqual(row.append_slots, [])
        self.assertEqual(row.borrowed_slots, row.row_slots)
        self.assertEqual(row.borrowed_append_slots_remaining, 1)
        self.assertEqual(len(row.row_slots), materialized_width + 1)
        self.assertEqual(row.row_slots[-1], first_full_slot)
        self.assertEqual(engine._token_pool_full_attention_slots, {})
        self.assertEqual(engine.metrics.token_pool_full_attention_row_invalidations, 0)
        before_reuse_allocated = engine.stats()["token_pool"]["allocated_token_slots"]

        req.num_computed_tokens += 1
        req.output_token_ids.append(88)
        reused_reservations = engine._token_pool_prepare_decode_batch(
            [req],
            persistent_full_attention_rows=True,
        )
        reused_context = engine._token_pool_decode_context(reused_reservations)
        self.assertIsNotNone(reused_context)
        reused_full = reused_context.metadata_by_layer_type["full_attention"]  # type: ignore[union-attr]
        reused_row = engine._token_pool_full_attention_rows["persist"]
        reused_full_slot = reused_reservations[0].full_attention_token_slot
        self.assertIsNotNone(reused_full_slot)
        self.assertEqual(reused_full_slot, reused_reservations[0].token_slot)
        self.assertEqual(reused_full.out_cache_loc.tolist(), [reused_full_slot])

        self.assertEqual(
            engine.stats()["token_pool"]["allocated_token_slots"],
            before_reuse_allocated,
        )
        self.assertIs(reused_row, row)
        self.assertEqual(reused_row.row_slots[-2], first_full_slot)
        self.assertEqual(reused_row.row_slots[-1], reused_full_slot)
        self.assertEqual(reused_row.append_slots, [])
        self.assertEqual(reused_row.borrowed_slots, reused_row.row_slots)
        self.assertEqual(reused_row.borrowed_append_slots_remaining, 0)
        self.assertEqual(reused_full.seq_lens.tolist(), [materialized_width + 2])
        for name, ptr in first_metadata_ptrs.items():
            self.assertEqual(int(getattr(reused_full, name).data_ptr()), ptr)
        self.assertEqual(
            reused_full.kv_indices[: materialized_width + 2].tolist(),
            reused_row.row_slots,
        )
        self.assertEqual(engine.metrics.token_pool_full_attention_row_rebuilds, 1)
        self.assertEqual(engine.metrics.token_pool_full_attention_row_reuses, 1)
        self.assertEqual(engine.metrics.token_pool_full_attention_row_appends, 1)

        engine._token_pool_discard_decode_reservations(reused_reservations)
        engine._token_pool_release_request(req.req_id)
        self.assertEqual(engine.stats()["token_pool"]["allocated_token_slots"], 0)

    def test_full_attention_prepare_rollback_rematerializes_released_rows(self) -> None:
        import torch

        from wkvm.core.request import Request
        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        engine, first_req, first_cache = self._make_full_attention_token_pool_fixture(
            req_id="first",
            num_slots=2,
            token_pool_capacity=128,
        )
        second_req = Request(
            prompt_token_ids=[4, 5, 6],
            max_new_tokens=4,
            req_id="second",
        )
        second_cache = NativeGemmaRoutedCache(first_cache.hf_config, engine.config)
        sliding_keys = torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2) + 40
        full_keys = torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2) + 80
        second_cache.update(sliding_keys, sliding_keys + 100, layer_idx=0)
        second_cache.update(full_keys, full_keys + 100, layer_idx=1)
        engine._caches[second_req.req_id] = second_cache
        engine._token_pool_commit_prefill_tokens(
            second_req,
            second_req.num_prompt_tokens,
            cache=second_cache,
            final_prefill=True,
        )
        second_req.num_computed_tokens = second_req.num_prompt_tokens
        second_req.output_token_ids.append(78)

        second_full_layer = second_cache.layers[1]

        def fail_second_row_write(*args, **kwargs):
            raise RuntimeError("forced second-row full-attention preparation failure")

        second_full_layer.write_materialized_readout_to_token_pool = (
            fail_second_row_write
        )
        prepared = engine._token_pool_prepare_decode_batch(
            [first_req, second_req],
            full_attention_kv_indices_padding_steps=1,
            persistent_full_attention_rows=True,
        )
        context = engine._token_pool_decode_context(prepared)

        self.assertIsNotNone(context)
        self.assertEqual(
            context.covered_layer_types,
            frozenset({"sliding_attention"}),
        )
        self.assertEqual(engine._token_pool_full_attention_rows, {})
        first_full_layer = first_cache.layers[1]
        self.assertIsNotNone(first_full_layer.keys)
        self.assertIsNotNone(first_full_layer.values)
        self.assertFalse(first_full_layer._dense_storage_released)

        merged, info = NativeGemmaRoutedCache.merge_padded_decode(
            [first_cache, second_cache],
            decode_steps=2,
            persistent=True,
            graph_static=True,
            token_pool_covered_layer_types=context.covered_layer_types,
        )
        self.assertEqual(info["merge"], "padded_valid_mask_concat")
        self.assertIsNotNone(merged.layers[1].keys)
        self.assertIsNotNone(merged.layers[1].values)

        engine._token_pool_discard_decode_reservations(prepared)
        engine._token_pool_release_request(first_req.req_id)
        engine._token_pool_release_request(second_req.req_id)
        self.assertEqual(engine.stats()["token_pool"]["allocated_token_slots"], 0)

    def test_persistent_full_attention_row_aliases_stable_suffix_only(self) -> None:
        engine, req, cache = self._make_full_attention_token_pool_fixture(
            req_id="persist-fallback"
        )
        materialized_width = cache.layers[1].materialized_tokens()
        allocated_before = engine.stats()["token_pool"]["allocated_token_slots"]

        reservations = engine._token_pool_prepare_decode_batch(
            [req],
            full_attention_kv_indices_padding_steps=5,
            persistent_full_attention_rows=True,
        )

        row = engine._token_pool_full_attention_rows[req.req_id]
        self.assertGreater(materialized_width + 6, engine.config.sliding_window)
        self.assertEqual(
            reservations[0].full_attention_token_slot,
            reservations[0].token_slot,
        )
        self.assertEqual(len(row.borrowed_slots), 3)
        self.assertEqual(row.borrowed_append_slots_remaining, 5)
        self.assertEqual(len(row.owned_slots), materialized_width - 2)
        self.assertEqual(
            engine.stats()["token_pool"]["allocated_token_slots"],
            allocated_before + materialized_width - 2,
        )

        engine._token_pool_discard_decode_reservations(reservations)
        engine._token_pool_release_request(req.req_id)
        self.assertEqual(engine.stats()["token_pool"]["allocated_token_slots"], 0)

    def test_paged_triton_env_does_not_page_align_full_attention_rows(self) -> None:
        engine, req, cache = self._make_full_attention_token_pool_fixture(
            req_id="paged-triton-flat",
            token_pool_capacity=128,
            token_pool_paged_block_size=4,
        )
        materialized_width = cache.layers[1].materialized_tokens()

        with unittest.mock.patch.dict(
            os.environ,
            {
                "WKVM_ENABLE_TOKEN_POOL_PAGED_TRITON": "1",
                "WKVM_ENABLE_TOKEN_POOL_PAGED_SPLIT_TRITON": "1",
                "WKVM_TOKEN_POOL_BUILD_PAGED_METADATA": "0",
            },
        ):
            reservations = engine._token_pool_prepare_decode_batch(
                [req],
                persistent_full_attention_rows=True,
            )
        context = engine._token_pool_decode_context(reservations)
        self.assertIsNotNone(context)
        full_metadata = context.metadata_by_layer_type["full_attention"]  # type: ignore[union-attr]
        self.assertEqual(full_metadata.seq_lens.tolist(), [materialized_width + 1])
        self.assertIsNone(context.paged_metadata_for_layer(1, "full_attention"))  # type: ignore[union-attr]
        self.assertNotIn(
            "full_attention",
            context.paged_metadata_by_layer_type or {},  # type: ignore[union-attr]
        )
        row = engine._token_pool_full_attention_rows["paged-triton-flat"]
        self.assertFalse(row.page_aligned)

        engine._token_pool_discard_decode_reservations(reservations)
        engine._token_pool_release_request(req.req_id)
        self.assertEqual(engine.stats()["token_pool"]["allocated_token_slots"], 0)

    def test_persistent_full_attention_rows_build_paged_metadata(self) -> None:
        import torch

        engine, req, cache = self._make_full_attention_token_pool_fixture(
            req_id="paged",
            token_pool_capacity=128,
            token_pool_paged_block_size=4,
        )
        full_layer = cache.layers[1]
        materialized_width = full_layer.materialized_tokens()

        with unittest.mock.patch.dict(
            os.environ,
            {"WKVM_TOKEN_POOL_BUILD_PAGED_METADATA": "1"},
        ):
            reservations = engine._token_pool_prepare_decode_batch(
                [req],
                persistent_full_attention_rows=True,
            )
        context = engine._token_pool_decode_context(reservations)
        self.assertIsNotNone(context)
        full_metadata = context.metadata_by_layer_type["full_attention"]  # type: ignore[union-attr]
        paged = context.paged_metadata_for_layer(1, "full_attention")  # type: ignore[union-attr]
        self.assertIsNotNone(paged)
        self.assertIs(paged, context.paged_metadata_by_layer_type["full_attention"])  # type: ignore[index,union-attr]
        self.assertEqual(paged.block_size, 4)
        self.assertEqual(paged.seq_lens.tolist(), full_metadata.seq_lens.tolist())
        self.assertEqual(
            paged.logical_seq_lens.tolist(),
            full_metadata.logical_seq_lens.tolist(),
        )
        self.assertEqual(paged.selected_start_positions.tolist(), [0])
        self.assertEqual(paged.out_cache_loc.tolist(), full_metadata.out_cache_loc.tolist())
        self.assertEqual(full_metadata.seq_lens.tolist(), [materialized_width + 1])

        flat_slots = full_metadata.kv_indices[: materialized_width + 1].tolist()
        reconstructed = []
        for offset in range(materialized_width + 1):
            block = int(paged.block_tables[0, offset // 4].item())
            reconstructed.append(block * 4 + (offset % 4))
        self.assertEqual(reconstructed, flat_slots)
        self.assertEqual(flat_slots[-1], reservations[0].full_attention_token_slot)

        decode_key = torch.tensor([[[301.0, 302.0]]])
        engine._token_kv_pool.set_kv(  # type: ignore[union-attr]
            1,
            [reservations[0].full_attention_token_slot],
            decode_key,
            decode_key + 100,
        )
        engine._token_pool_commit_decode_reservations(reservations)
        req.num_computed_tokens += 1
        req.output_token_ids.append(88)

        with unittest.mock.patch.dict(
            os.environ,
            {"WKVM_TOKEN_POOL_BUILD_PAGED_METADATA": "1"},
        ):
            reused_reservations = engine._token_pool_prepare_decode_batch(
                [req],
                persistent_full_attention_rows=True,
            )
        reused_context = engine._token_pool_decode_context(reused_reservations)
        self.assertIsNotNone(reused_context)
        reused_full = reused_context.metadata_by_layer_type["full_attention"]  # type: ignore[union-attr]
        reused_paged = reused_context.paged_metadata_for_layer(1, "full_attention")  # type: ignore[union-attr]
        self.assertIsNotNone(reused_paged)
        self.assertEqual(reused_paged.seq_lens.tolist(), reused_full.seq_lens.tolist())
        reused_slots = reused_full.kv_indices[: materialized_width + 2].tolist()
        reconstructed = []
        for offset in range(materialized_width + 2):
            block = int(reused_paged.block_tables[0, offset // 4].item())
            reconstructed.append(block * 4 + (offset % 4))
        self.assertEqual(reconstructed, reused_slots)

        engine._token_pool_discard_decode_reservations(reused_reservations)
        engine._token_pool_release_request(req.req_id)
        self.assertEqual(engine.stats()["token_pool"]["allocated_token_slots"], 0)

    def test_persistent_full_attention_paged_metadata_pads_block_table_width(self) -> None:
        import torch

        engine, req, cache = self._make_full_attention_token_pool_fixture(
            req_id="paged-padding",
            token_pool_capacity=128,
            token_pool_paged_block_size=4,
        )

        with unittest.mock.patch.dict(
            os.environ,
            {"WKVM_TOKEN_POOL_BUILD_PAGED_METADATA": "1"},
        ):
            reservations = engine._token_pool_prepare_decode_batch(
                [req],
                full_attention_kv_indices_padding_steps=2,
                persistent_full_attention_rows=True,
            )
        context = engine._token_pool_decode_context(reservations)
        self.assertIsNotNone(context)
        full_metadata = context.metadata_by_layer_type["full_attention"]  # type: ignore[union-attr]
        paged = context.paged_metadata_for_layer(1, "full_attention")  # type: ignore[union-attr]
        self.assertIsNotNone(paged)
        expected_width = (int(full_metadata.max_seq_len) + paged.block_size - 1) // paged.block_size
        self.assertEqual(tuple(paged.block_tables.shape), (1, expected_width))
        first_signature = engine._token_pool_decode_shape_signature(context)[
            "paged_metadata_by_layer_type"
        ]["full_attention"]

        decode_key = torch.tensor([[[401.0, 402.0]]])
        engine._token_kv_pool.set_kv(  # type: ignore[union-attr]
            1,
            [reservations[0].full_attention_token_slot],
            decode_key,
            decode_key + 100,
        )
        engine._token_pool_commit_decode_reservations(reservations)
        req.num_computed_tokens += 1
        req.output_token_ids.append(88)

        with unittest.mock.patch.dict(
            os.environ,
            {"WKVM_TOKEN_POOL_BUILD_PAGED_METADATA": "1"},
        ):
            reused_reservations = engine._token_pool_prepare_decode_batch(
                [req],
                full_attention_kv_indices_padding_steps=1,
                persistent_full_attention_rows=True,
            )
        reused_context = engine._token_pool_decode_context(reused_reservations)
        self.assertIsNotNone(reused_context)
        reused_full = reused_context.metadata_by_layer_type["full_attention"]  # type: ignore[union-attr]
        reused_paged = reused_context.paged_metadata_for_layer(1, "full_attention")  # type: ignore[union-attr]
        self.assertIsNotNone(reused_paged)
        self.assertEqual(reused_full.max_seq_len, full_metadata.max_seq_len)
        self.assertEqual(reused_paged.max_seq_len, paged.max_seq_len)
        self.assertEqual(tuple(reused_paged.block_tables.shape), tuple(paged.block_tables.shape))
        self.assertEqual(
            engine._token_pool_decode_shape_signature(reused_context)[
                "paged_metadata_by_layer_type"
            ]["full_attention"],
            first_signature,
        )

        engine._token_pool_discard_decode_reservations(reused_reservations)
        engine._token_pool_release_request(req.req_id)
        self.assertEqual(engine.stats()["token_pool"]["allocated_token_slots"], 0)

    def test_flush_padded_decode_group_clears_persistent_full_attention_row(self) -> None:
        engine, req, cache = self._make_full_attention_token_pool_fixture(
            req_id="flush"
        )
        materialized_width = cache.layers[1].materialized_tokens()

        reservations = engine._token_pool_prepare_decode_batch(
            [req],
            persistent_full_attention_rows=True,
        )
        allocated_with_row = engine.stats()["token_pool"]["allocated_token_slots"]
        self.assertIn("flush", engine._token_pool_full_attention_rows)

        engine._flush_padded_decode_group((req.req_id,))

        self.assertEqual(engine._token_pool_full_attention_rows, {})
        self.assertEqual(engine._token_pool_full_attention_slots, {})
        self.assertEqual(
            engine.stats()["token_pool"]["allocated_token_slots"],
            allocated_with_row,
        )
        self.assertEqual(engine.metrics.token_pool_full_attention_row_invalidations, 0)

        engine._token_pool_discard_decode_reservations(reservations)
        self.assertEqual(
            engine.stats()["token_pool"]["allocated_token_slots"],
            16,
        )
        engine._token_pool_release_request(req.req_id)
        self.assertEqual(engine.stats()["token_pool"]["allocated_token_slots"], 0)

    def test_token_pool_commit_mirrors_full_decode_kv_to_routed_cache(self) -> None:
        import torch
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config
        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        class FakeNativeTokenPoolModel:
            wkvm_no_hf_transformer_forward = True

            def __init__(self) -> None:
                self._param = torch.empty((), dtype=torch.float32)
                self.config = SimpleNamespace(
                    num_attention_heads=2,
                    layer_types=("sliding_attention", "full_attention"),
                )
                metas = [
                    SimpleNamespace(
                        layer_type="sliding_attention",
                        is_kv_shared_layer=False,
                        num_key_value_groups=2,
                        head_dim=2,
                    ),
                    SimpleNamespace(
                        layer_type="full_attention",
                        is_kv_shared_layer=False,
                        num_key_value_groups=2,
                        head_dim=2,
                    ),
                ]
                self.text_prefix = SimpleNamespace(
                    layers=[
                        SimpleNamespace(layer_idx=idx, attn_meta=meta)
                        for idx, meta in enumerate(metas)
                    ]
                )

            def parameters(self):
                return iter([self._param])

        hf_cfg = SimpleNamespace(
            num_hidden_layers=2,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention", "full_attention"),
            sliding_window=8,
        )
        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=2,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention", "full_attention"),
            sink_tokens=1,
            ring_tokens=8,
            pending_tokens=8,
            routed_slots=2,
            reps_per_slot=1,
            span_budget_tokens=4,
            max_span_tokens=3,
            sliding_window=8,
        )
        engine = GemmaNativeEngine(
            model=FakeNativeTokenPoolModel(),
            config=cfg,
            num_slots=1,
            enable_token_pool_attention=True,
            token_pool_max_context_len=8,
            token_pool_capacity=32,
        )
        req = Request(prompt_token_ids=[1, 2, 3], max_new_tokens=2, req_id="mirror")
        cache = NativeGemmaRoutedCache(hf_cfg, cfg)
        sliding_keys = torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2)
        full_keys = torch.arange(20, 26, dtype=torch.float32).reshape(1, 1, 3, 2)
        cache.update(sliding_keys, sliding_keys + 100, layer_idx=0)
        cache.update(full_keys, full_keys + 100, layer_idx=1)
        engine._caches[req.req_id] = cache
        engine._token_pool_commit_prefill_tokens(
            req,
            req.num_prompt_tokens,
            cache=cache,
            final_prefill=True,
        )
        req.num_computed_tokens = req.num_prompt_tokens
        req.output_token_ids.append(88)
        reservations = engine._token_pool_prepare_decode_batch([req])
        full_layer = cache.layers[1]
        previous_length = full_layer.cumulative_length
        decode_key = torch.tensor([[[301.0, 302.0]]])
        decode_value = decode_key + 100
        engine._token_kv_pool.set_kv(  # type: ignore[union-attr]
            1,
            [reservations[0].token_slot],
            decode_key,
            decode_value,
        )
        original_commit_decode_token = full_layer.commit_decode_token

        def commit_decode_token_in_inference_mode(key_states, value_states):
            self.assertTrue(torch.is_inference_mode_enabled())
            return original_commit_decode_token(key_states, value_states)

        full_layer.commit_decode_token = commit_decode_token_in_inference_mode

        engine._token_pool_commit_decode_reservations(reservations)

        self.assertEqual(full_layer.cumulative_length, previous_length + 1)
        self.assertIsNone(full_layer.keys)
        self.assertIsNone(full_layer.values)
        self.assertTrue(full_layer._dense_storage_released)
        self.assertTrue(
            torch.equal(
                full_layer._ring_k[:, :, -1:, :],
                decode_key.permute(1, 0, 2).unsqueeze(0),
            )
        )
        self.assertTrue(
            torch.equal(
                full_layer._ring_v[:, :, -1:, :],
                decode_value.permute(1, 0, 2).unsqueeze(0),
            )
        )
        self.assertTrue(full_layer.restore_dense_materialized_storage())
        self.assertTrue(
            torch.equal(
                full_layer.keys[:, :, -1:, :],
                decode_key.permute(1, 0, 2).unsqueeze(0),
            )
        )
        self.assertEqual(engine._token_pool_full_attention_slots, {})
        self.assertEqual(engine.stats()["token_pool"]["allocated_token_slots"], 16)

    def test_token_pool_attention_backfills_only_available_sliding_tail(self) -> None:
        import torch
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        class FakeNativeTokenPoolModel:
            wkvm_no_hf_transformer_forward = True

            def __init__(self) -> None:
                self._param = torch.empty((), dtype=torch.float32)
                self.config = SimpleNamespace(num_attention_heads=2)
                attn = SimpleNamespace(
                    layer_type="sliding_attention",
                    is_kv_shared_layer=False,
                    num_key_value_groups=2,
                    head_dim=4,
                )
                self.text_prefix = SimpleNamespace(
                    layers=[SimpleNamespace(layer_idx=0, attn_meta=attn)]
                )

            def parameters(self):
                return iter([self._param])

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=4,
        )
        engine = GemmaNativeEngine(
            model=FakeNativeTokenPoolModel(),
            config=cfg,
            num_slots=1,
            enable_token_pool_attention=True,
            token_pool_max_context_len=8,
            token_pool_capacity=32,
        )
        req = Request(prompt_token_ids=[1, 2, 3, 4, 5], max_new_tokens=1, req_id="tail")
        keys = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
        values = keys + 100
        cache = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    is_sliding=True,
                    keys=keys,
                    values=values,
                )
            ]
        )

        engine._token_pool_commit_prefill_tokens(req, 5, cache=cache)

        self.assertEqual(engine._token_table.slots_for("tail").tolist(), [-1, -1, 2, 3, 4])
        metadata = engine._token_table.build_decode_metadata(
            [engine._token_pool_req_slots["tail"]],
            sliding_window=4,
            allow_padding=True,
        )
        self.assertEqual(metadata.logical_seq_lens.tolist(), [5])
        self.assertEqual(metadata.seq_lens.tolist(), [3])
        self.assertEqual(metadata.kv_indices.tolist(), [2, 3, 4])
        gathered_k, gathered_v = engine._token_kv_pool.gather_kv(0, [2, 3, 4])
        self.assertTrue(torch.equal(gathered_k, keys[0].permute(1, 0, 2)))
        self.assertTrue(torch.equal(gathered_v, values[0].permute(1, 0, 2)))

    def test_token_pool_final_prefill_backfill_releases_source_sliding_layer(self) -> None:
        import torch
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        class FakeNativeTokenPoolModel:
            wkvm_no_hf_transformer_forward = True

            def __init__(self) -> None:
                self._param = torch.empty((), dtype=torch.float32)
                self.config = SimpleNamespace(num_attention_heads=2)
                attn = SimpleNamespace(
                    layer_type="sliding_attention",
                    is_kv_shared_layer=False,
                    num_key_value_groups=2,
                    head_dim=4,
                )
                self.text_prefix = SimpleNamespace(
                    layers=[SimpleNamespace(layer_idx=0, attn_meta=attn)]
                )

            def parameters(self):
                return iter([self._param])

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=4,
        )
        engine = GemmaNativeEngine(
            model=FakeNativeTokenPoolModel(),
            config=cfg,
            num_slots=1,
            enable_token_pool_attention=True,
            token_pool_max_context_len=4,
            token_pool_capacity=32,
        )
        engine._token_kv_pool = engine._build_token_kv_pool(
            capacity=32,
            defer_buffer_allocation=True,
        )
        engine._token_slot_allocator = engine._token_kv_pool
        req = Request(prompt_token_ids=[1, 2, 3], max_new_tokens=1, req_id="final")
        keys = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)
        values = keys + 100
        layer = SimpleNamespace(
            is_sliding=True,
            keys=keys,
            values=values,
            _dense_storage_released=False,
        )
        cache = SimpleNamespace(layers=[layer])

        self.assertEqual(engine._token_kv_pool.state_bytes(), 0)
        engine._token_pool_commit_prefill_tokens(
            req,
            3,
            cache=cache,
            final_prefill=True,
        )

        self.assertIsNone(layer.keys)
        self.assertIsNone(layer.values)
        self.assertTrue(layer._dense_storage_released)
        gathered_k, gathered_v = engine._token_kv_pool.gather_kv(0, [0, 1, 2])
        self.assertTrue(torch.equal(gathered_k, keys[0].permute(1, 0, 2)))
        self.assertTrue(torch.equal(gathered_v, values[0].permute(1, 0, 2)))

    def test_token_pool_attention_retains_only_sliding_window_slots(self) -> None:
        import torch
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        class FakeNativeTokenPoolModel:
            wkvm_no_hf_transformer_forward = True

            def __init__(self) -> None:
                self._param = torch.empty((), dtype=torch.float32)
                self.config = SimpleNamespace(num_attention_heads=2)
                attn = SimpleNamespace(
                    layer_type="sliding_attention",
                    is_kv_shared_layer=False,
                    num_key_value_groups=2,
                    head_dim=4,
                )
                self.text_prefix = SimpleNamespace(
                    layers=[SimpleNamespace(layer_idx=0, attn_meta=attn)]
                )

            def parameters(self):
                return iter([self._param])

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=4,
        )
        engine = GemmaNativeEngine(
            model=FakeNativeTokenPoolModel(),
            config=cfg,
            num_slots=1,
            enable_token_pool_attention=True,
            token_pool_max_context_len=8,
            token_pool_capacity=32,
        )
        req = Request(prompt_token_ids=list(range(8)), max_new_tokens=1, req_id="win")
        cache = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    is_sliding=True,
                    keys=torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4),
                    values=torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) + 100,
                )
            ]
        )

        engine._token_pool_commit_prefill_tokens(req, 4, cache=cache)
        req.num_computed_tokens = 4
        self.assertEqual(engine.stats()["token_pool"]["allocated_token_slots"], 16)

        engine._token_pool_commit_prefill_tokens(req, 4, cache=cache)
        self.assertEqual(engine._token_table.length("win"), 8)
        self.assertEqual(engine._token_table.slots_for("win").tolist(), [-1, -1, -1, -1, 4, 5, 6, 7])
        self.assertEqual(engine.stats()["token_pool"]["allocated_token_slots"], 16)

    def test_token_pool_attention_reclaims_expired_page_blocks(self) -> None:
        import torch
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        class FakeNativeTokenPoolModel:
            wkvm_no_hf_transformer_forward = True

            def __init__(self) -> None:
                self._param = torch.empty((), dtype=torch.float32)
                self.config = SimpleNamespace(num_attention_heads=2)
                attn = SimpleNamespace(
                    layer_type="sliding_attention",
                    is_kv_shared_layer=False,
                    num_key_value_groups=2,
                    head_dim=4,
                )
                self.text_prefix = SimpleNamespace(
                    layers=[SimpleNamespace(layer_idx=0, attn_meta=attn)]
                )

            def parameters(self):
                return iter([self._param])

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=4,
        )
        engine = GemmaNativeEngine(
            model=FakeNativeTokenPoolModel(),
            config=cfg,
            num_slots=1,
            enable_token_pool_attention=True,
            token_pool_max_context_len=16,
            token_pool_capacity=8,
            token_pool_paged_block_size=4,
        )
        req = Request(prompt_token_ids=[1, 2, 3, 4], max_new_tokens=5, req_id="pages")
        keys = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        cache = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    is_sliding=True,
                    keys=keys,
                    values=keys + 100,
                )
            ]
        )

        engine._token_pool_commit_prefill_tokens(req, 4, cache=cache)
        req.num_computed_tokens = 4
        self.assertEqual(engine.stats()["token_pool"]["allocated_token_slots"], 4)

        for step in range(5):
            req.output_token_ids.append(100 + step)
            reservations = engine._token_pool_prepare_decode_batch([req])
            engine._token_pool_commit_decode_reservations(reservations)
            req.num_computed_tokens += 1

        req_slot = engine._token_pool_req_slots[req.req_id]
        page_table_tensor = engine._token_pool_page_table_tensor
        self.assertIsNotNone(page_table_tensor)
        self.assertEqual(page_table_tensor[req_slot, :3].tolist(), [-1, 1, 0])
        self.assertLessEqual(
            engine.stats()["token_pool"]["allocated_token_slots"],
            8,
        )
        self.assertEqual(
            engine._token_table.slots_for("pages").tolist(),
            [-1, -1, -1, -1, -1, 5, 6, 7, 0],
        )

        engine._token_pool_release_request(req.req_id)
        self.assertEqual(engine.stats()["token_pool"]["allocated_token_slots"], 0)

    def test_decode_prepare_failure_rolls_back_page_owned_decode_block(self) -> None:
        import torch
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        class FakeNativeTokenPoolModel:
            wkvm_no_hf_transformer_forward = True

            def __init__(self) -> None:
                self._param = torch.empty((), dtype=torch.float32)
                self.config = SimpleNamespace(num_attention_heads=2)
                attn = SimpleNamespace(
                    layer_type="sliding_attention",
                    is_kv_shared_layer=False,
                    num_key_value_groups=2,
                    head_dim=4,
                )
                self.text_prefix = SimpleNamespace(
                    layers=[SimpleNamespace(layer_idx=0, attn_meta=attn)]
                )

            def parameters(self):
                return iter([self._param])

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=4,
        )
        engine = GemmaNativeEngine(
            model=FakeNativeTokenPoolModel(),
            config=cfg,
            num_slots=1,
            enable_token_pool_attention=True,
            token_pool_max_context_len=8,
            token_pool_capacity=8,
            token_pool_paged_block_size=4,
        )
        req = Request(prompt_token_ids=[1, 2, 3, 4], max_new_tokens=1, req_id="fail")
        keys = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
        cache = SimpleNamespace(
            layers=[
                SimpleNamespace(
                    is_sliding=True,
                    keys=keys,
                    values=keys + 100,
                )
            ]
        )

        engine._token_pool_commit_prefill_tokens(req, 4, cache=cache)
        req.num_computed_tokens = 4
        req.output_token_ids.append(99)
        req_slot = engine._token_pool_req_slots[req.req_id]
        page_table_tensor = engine._token_pool_page_table_tensor
        self.assertIsNotNone(page_table_tensor)
        self.assertEqual(engine.stats()["token_pool"]["allocated_token_slots"], 4)
        self.assertEqual(engine._token_pool_page_tables[req.req_id], {0: 0})
        self.assertEqual(
            engine._token_pool_page_owned_slots[req.req_id],
            {0, 1, 2, 3},
        )
        self.assertEqual(page_table_tensor[req_slot, :1].tolist(), [0])

        original_prepare_layer_metadata = (
            engine._token_pool_prepare_layer_decode_metadata
        )

        def raise_after_decode_page_alloc(*args, **kwargs):
            raise RuntimeError("forced metadata failure")

        engine._token_pool_prepare_layer_decode_metadata = (  # type: ignore[method-assign]
            raise_after_decode_page_alloc
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "forced metadata failure"):
                engine._token_pool_prepare_decode_batch([req])
        finally:
            engine._token_pool_prepare_layer_decode_metadata = (  # type: ignore[method-assign]
                original_prepare_layer_metadata
            )

        self.assertEqual(engine.stats()["token_pool"]["allocated_token_slots"], 4)
        self.assertEqual(engine._token_table.length(req_slot), 4)
        self.assertEqual(
            engine._token_table.slots_for(req.req_id).tolist(),
            [0, 1, 2, 3],
        )
        self.assertEqual(engine._token_pool_page_tables[req.req_id], {0: 0})
        self.assertEqual(
            engine._token_pool_page_owned_slots[req.req_id],
            {0, 1, 2, 3},
        )
        self.assertEqual(page_table_tensor[req_slot, :2].tolist(), [0, -1])

    def test_token_pool_clear_prefix_reclaims_page_blocks_without_dropped_slots(self) -> None:
        import torch
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        class FakeNativeTokenPoolModel:
            wkvm_no_hf_transformer_forward = True

            def __init__(self) -> None:
                self._param = torch.empty((), dtype=torch.float32)
                self.config = SimpleNamespace(num_attention_heads=2)
                attn = SimpleNamespace(
                    layer_type="sliding_attention",
                    is_kv_shared_layer=False,
                    num_key_value_groups=2,
                    head_dim=4,
                )
                self.text_prefix = SimpleNamespace(
                    layers=[SimpleNamespace(layer_idx=0, attn_meta=attn)]
                )

            def parameters(self):
                return iter([self._param])

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=4,
        )
        engine = GemmaNativeEngine(
            model=FakeNativeTokenPoolModel(),
            config=cfg,
            num_slots=1,
            enable_token_pool_attention=True,
            token_pool_max_context_len=8,
            token_pool_capacity=4,
            token_pool_paged_block_size=4,
        )
        req = Request(prompt_token_ids=[1, 2, 3, 4], max_new_tokens=1, req_id="empty")
        engine._token_pool_admit_request(req)
        req_slot = engine._token_pool_req_slots[req.req_id]
        _, slot_ids = engine._token_pool_alloc_page_aligned_slots(req.req_id, 0, 1)
        self.assertEqual(slot_ids, [0])
        engine._token_table.append_slots(req_slot, [engine._token_table.padding_token] * 4)

        page_table_tensor = engine._token_pool_page_table_tensor
        self.assertIsNotNone(page_table_tensor)
        self.assertEqual(page_table_tensor[req_slot, :1].tolist(), [0])
        self.assertEqual(engine.stats()["token_pool"]["allocated_token_slots"], 4)

        engine._token_pool_clear_prefix(req.req_id, req_slot, 4)

        self.assertEqual(page_table_tensor[req_slot, :1].tolist(), [-1])
        self.assertEqual(engine._token_pool_page_tables[req.req_id], {})
        self.assertEqual(engine._token_pool_page_owned_slots[req.req_id], set())
        self.assertEqual(engine.stats()["token_pool"]["allocated_token_slots"], 0)

    def test_decode_batch_fallback_discards_token_pool_reservations(self) -> None:
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=2,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention", "sliding_attention"),
        )
        engine = GemmaNativeEngine(
            model=FakeModel(),
            config=cfg,
            num_slots=2,
            enable_token_pool_metadata=True,
            scheduler_config=SchedulerConfig(
                max_tokens_per_step=16,
                max_running_requests=2,
                max_tokens_per_request_per_step=8,
            ),
        )
        runner = FakeDistinctBatchRunner()
        engine.runner = runner  # type: ignore[assignment]
        reqs = [
            Request(prompt_token_ids=[1, 2, 3], max_new_tokens=3, req_id="a"),
            Request(prompt_token_ids=[4, 5, 6], max_new_tokens=3, req_id="b"),
        ]
        for req in reqs:
            engine.add_request(req)

        engine.step()
        engine.step()

        self.assertEqual(runner.decode_batch_calls, [([101, 102], [3, 3])])
        self.assertEqual(runner.decode_step_calls, 2)
        self.assertEqual(engine.stats()["token_pool"]["allocated_token_slots"], 8)
        table = engine._token_table
        self.assertIsNotNone(table)
        self.assertEqual(table.slots_for("a").tolist(), [0, 1, 2, 6])
        self.assertEqual(table.slots_for("b").tolist(), [3, 4, 5, 7])
        self.assertEqual(engine.metrics.token_pool_decode_metadata_batches, 3)
        self.assertEqual(engine.metrics.token_pool_decode_metadata_rows, 4)

    def test_persistent_exact_decode_reuses_group_until_finish(self) -> None:
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
        )
        engine = GemmaNativeEngine(
            model=FakeModel(),
            config=cfg,
            num_slots=2,
            scheduler_config=SchedulerConfig(
                max_tokens_per_step=16,
                max_running_requests=2,
                max_tokens_per_request_per_step=8,
            ),
        )
        runner = FakePersistentExactBatchRunner()
        engine.runner = runner  # type: ignore[assignment]

        reqs = [
            Request(prompt_token_ids=[1, 2, 3], max_new_tokens=4, req_id="a"),
            Request(prompt_token_ids=[4, 5, 6], max_new_tokens=4, req_id="b"),
        ]
        for req in reqs:
            engine.add_request(req)

        engine.step()
        self.assertEqual([req.output_token_ids for req in reqs], [[101], [102]])

        engine.step()
        self.assertEqual(runner.persistent_starts, [([101, 102], [3, 3])])
        self.assertEqual(runner.persistent_reuses, [])
        self.assertEqual(engine.metrics.persistent_exact_decode_starts, 1)
        self.assertEqual(engine.metrics.persistent_exact_decode_splits, 0)

        engine.step()
        self.assertEqual(runner.persistent_reuses, [([300, 301], [4, 4])])
        self.assertEqual(engine.metrics.persistent_exact_decode_reuses, 1)
        self.assertEqual(engine.metrics.persistent_exact_decode_splits, 0)

        finished = engine.step()
        self.assertEqual({req.req_id for req in finished}, {"a", "b"})
        self.assertEqual(runner.decode_batch_calls, [])
        self.assertEqual(runner.decode_step_calls, 0)
        self.assertEqual(engine.metrics.decode_model_calls, 3)
        self.assertEqual(engine.metrics.batched_decode_model_calls, 3)
        self.assertEqual(engine.metrics.exact_decode_batch_rows, 6)
        self.assertEqual(engine.metrics.persistent_exact_decode_rows, 6)
        self.assertEqual(engine.metrics.persistent_exact_decode_starts, 1)
        self.assertEqual(engine.metrics.persistent_exact_decode_reuses, 2)
        self.assertEqual(engine.metrics.persistent_exact_decode_splits, 1)
        self.assertFalse(engine._persistent_exact_decode_groups)
        self.assertTrue(all(req.status.is_finished for req in reqs))

    def test_persistent_padded_decode_reuses_group_until_finish(self) -> None:
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
        )
        engine = GemmaNativeEngine(
            model=FakeModel(),
            config=cfg,
            num_slots=2,
            scheduler_config=SchedulerConfig(
                max_tokens_per_step=16,
                max_running_requests=2,
                max_tokens_per_request_per_step=8,
            ),
            persistent_exact_decode=False,
            persistent_padded_decode_steps=3,
        )
        runner = FakePersistentPaddedBatchRunner()
        engine.runner = runner  # type: ignore[assignment]

        reqs = [
            Request(prompt_token_ids=[1, 2, 3], max_new_tokens=4, req_id="a"),
            Request(prompt_token_ids=[4, 5, 6], max_new_tokens=4, req_id="b"),
        ]
        for req in reqs:
            engine.add_request(req)

        engine.step()
        self.assertEqual([req.output_token_ids for req in reqs], [[101], [102]])

        engine.step()
        self.assertEqual(runner.persistent_padded_starts, [([101, 102], [3, 3], 3)])
        self.assertEqual(runner.persistent_padded_reuses, [])
        self.assertEqual(engine.metrics.persistent_padded_decode_starts, 1)
        self.assertEqual(engine.metrics.persistent_padded_decode_splits, 0)

        engine.step()
        self.assertEqual(runner.persistent_padded_reuses, [([500, 501], [4, 4])])
        self.assertEqual(engine.metrics.persistent_padded_decode_reuses, 1)
        self.assertEqual(engine.metrics.persistent_padded_decode_splits, 0)

        finished = engine.step()
        self.assertEqual({req.req_id for req in finished}, {"a", "b"})
        self.assertEqual(
            runner.persistent_padded_reuses,
            [([500, 501], [4, 4]), ([610, 611], [5, 5])],
        )
        self.assertEqual(runner.decode_batch_calls, [])
        self.assertEqual(runner.decode_step_calls, 0)
        self.assertEqual(engine.metrics.decode_model_calls, 3)
        self.assertEqual(engine.metrics.batched_decode_model_calls, 3)
        self.assertEqual(engine.metrics.padded_decode_batch_rows, 6)
        self.assertEqual(engine.metrics.persistent_padded_decode_rows, 6)
        self.assertEqual(engine.metrics.persistent_padded_decode_starts, 1)
        self.assertEqual(engine.metrics.persistent_padded_decode_reuses, 2)
        self.assertEqual(engine.metrics.persistent_padded_decode_splits, 1)
        self.assertEqual(engine.metrics.padded_decode_temp_bytes, 1000)
        self.assertFalse(engine._persistent_padded_decode_groups)
        self.assertEqual(runner.merged_cache.remaining_capacity_calls, 2)
        self.assertEqual(runner.merged_cache.commit_count, 1)
        self.assertTrue(all(req.status.is_finished for req in reqs))

    def test_persistent_padded_token_pool_graph_shape_metrics(self) -> None:
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
        )
        engine = GemmaNativeEngine(
            model=FakeModel(),
            config=cfg,
            num_slots=2,
            scheduler_config=SchedulerConfig(
                max_tokens_per_step=16,
                max_running_requests=2,
                max_tokens_per_request_per_step=8,
            ),
            persistent_exact_decode=False,
            persistent_padded_decode_steps=3,
            persistent_padded_decode_cuda_graph=True,
        )
        runner = FakePersistentPaddedBatchRunner()
        engine.runner = runner  # type: ignore[assignment]
        contexts = [
            fake_token_pool_decode_context(kv_indices=4),
            fake_token_pool_decode_context(kv_indices=4),
            fake_token_pool_decode_context(kv_indices=5),
        ]
        engine._token_pool_prepare_decode_batch = lambda reqs, **kwargs: [object()]  # type: ignore[method-assign]
        engine._token_pool_decode_context = lambda reservations: contexts.pop(0)  # type: ignore[method-assign]
        engine._token_pool_commit_decode_reservations = lambda reservations: None  # type: ignore[method-assign]

        reqs = [
            Request(prompt_token_ids=[1, 2, 3], max_new_tokens=4, req_id="a"),
            Request(prompt_token_ids=[4, 5, 6], max_new_tokens=4, req_id="b"),
        ]
        for req in reqs:
            engine.add_request(req)

        engine.step()
        engine.step()
        engine.step()
        engine.step()

        self.assertEqual(engine.metrics.token_pool_decode_graph_candidate_batches, 3)
        self.assertEqual(engine.metrics.token_pool_decode_graph_static_shape_starts, 1)
        self.assertEqual(engine.metrics.token_pool_decode_graph_static_shape_reuses, 1)
        self.assertEqual(engine.metrics.token_pool_decode_graph_shape_mismatches, 1)
        self.assertEqual(
            engine.metrics.token_pool_decode_graph_shape_mismatch_reasons,
            {
                "metadata_by_layer_type.sliding_attention.kv_indices": 1,
                "metadata_by_layer_id.0.kv_indices": 1,
            },
        )
        self.assertFalse(engine._persistent_padded_token_pool_decode_signatures)

    def test_persistent_padded_graph_metadata_fallback_records_shape_mismatch(self) -> None:
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
        )
        engine = GemmaNativeEngine(
            model=FakeModel(),
            config=cfg,
            num_slots=2,
            scheduler_config=SchedulerConfig(
                max_tokens_per_step=16,
                max_running_requests=2,
                max_tokens_per_request_per_step=8,
            ),
            persistent_exact_decode=False,
            persistent_padded_decode_steps=3,
            persistent_padded_decode_cuda_graph=True,
        )
        runner = FakeGraphMismatchPersistentPaddedBatchRunner()
        engine.runner = runner  # type: ignore[assignment]
        contexts = [
            fake_token_pool_decode_context(kv_indices=4),
            fake_token_pool_decode_context(kv_indices=5),
            fake_token_pool_decode_context(kv_indices=5),
        ]
        discarded: list[list[object]] = []
        engine._token_pool_prepare_decode_batch = lambda reqs, **kwargs: [object()]  # type: ignore[method-assign]
        engine._token_pool_decode_context = lambda reservations: contexts.pop(0)  # type: ignore[method-assign]
        engine._token_pool_commit_decode_reservations = lambda reservations: None  # type: ignore[method-assign]
        engine._token_pool_discard_decode_reservations = (  # type: ignore[method-assign]
            lambda reservations: discarded.append(list(reservations))
        )

        reqs = [
            Request(prompt_token_ids=[1, 2, 3], max_new_tokens=4, req_id="a"),
            Request(prompt_token_ids=[4, 5, 6], max_new_tokens=4, req_id="b"),
        ]
        for req in reqs:
            engine.add_request(req)

        engine.step()
        engine.step()
        engine.step()

        self.assertEqual(engine.metrics.token_pool_decode_graph_candidate_batches, 2)
        self.assertEqual(engine.metrics.token_pool_decode_graph_static_shape_starts, 1)
        self.assertEqual(engine.metrics.token_pool_decode_graph_static_shape_reuses, 0)
        self.assertEqual(engine.metrics.token_pool_decode_graph_shape_mismatches, 1)
        self.assertEqual(
            engine.metrics.token_pool_decode_graph_shape_mismatch_reasons,
            {
                "metadata_by_layer_type.sliding_attention.kv_indices": 1,
                "metadata_by_layer_id.0.kv_indices": 1,
            },
        )
        self.assertEqual(len(discarded), 1)
        self.assertEqual(len(runner.decode_batch_calls), 1)
        self.assertFalse(engine._persistent_padded_token_pool_decode_signatures)

    def test_persistent_padded_token_pool_graph_shape_metrics_skip_without_graph_recording(self) -> None:
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        with unittest.mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("WKVM_TOKEN_POOL_RECORD_GRAPH_SIGNATURES", None)
            cfg = gemma4_e4b_routed_span_config(
                num_hidden_layers=1,
                num_kv_shared_layers=0,
                layer_types=("sliding_attention",),
            )
            engine = GemmaNativeEngine(
                model=FakeModel(),
                config=cfg,
                num_slots=2,
                scheduler_config=SchedulerConfig(
                    max_tokens_per_step=16,
                    max_running_requests=2,
                    max_tokens_per_request_per_step=8,
                ),
                persistent_exact_decode=False,
                persistent_padded_decode_steps=3,
            )
        runner = FakePersistentPaddedBatchRunner()
        engine.runner = runner  # type: ignore[assignment]
        contexts = [
            fake_token_pool_decode_context(kv_indices=4),
            fake_token_pool_decode_context(kv_indices=4),
            fake_token_pool_decode_context(kv_indices=5),
        ]
        engine._token_pool_prepare_decode_batch = lambda reqs, **kwargs: [object()]  # type: ignore[method-assign]
        engine._token_pool_decode_context = lambda reservations: contexts.pop(0)  # type: ignore[method-assign]
        engine._token_pool_commit_decode_reservations = lambda reservations: None  # type: ignore[method-assign]

        reqs = [
            Request(prompt_token_ids=[1, 2, 3], max_new_tokens=4, req_id="a"),
            Request(prompt_token_ids=[4, 5, 6], max_new_tokens=4, req_id="b"),
        ]
        for req in reqs:
            engine.add_request(req)

        engine.step()
        engine.step()
        engine.step()
        engine.step()

        self.assertFalse(engine.record_token_pool_decode_graph_signatures)
        self.assertEqual(engine.metrics.token_pool_decode_graph_candidate_batches, 0)
        self.assertEqual(engine.metrics.token_pool_decode_graph_static_shape_starts, 0)
        self.assertEqual(engine.metrics.token_pool_decode_graph_static_shape_reuses, 0)
        self.assertEqual(engine.metrics.token_pool_decode_graph_shape_mismatches, 0)
        self.assertEqual(engine.metrics.token_pool_decode_graph_shape_mismatch_reasons, {})
        self.assertFalse(engine._persistent_padded_token_pool_decode_signatures)

    def test_single_row_cuda_graph_persistent_padded_decode_reuses_group(self) -> None:
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
        )
        engine = GemmaNativeEngine(
            model=FakeModel(),
            config=cfg,
            num_slots=1,
            scheduler_config=SchedulerConfig(
                max_tokens_per_step=16,
                max_running_requests=1,
                max_tokens_per_request_per_step=8,
            ),
            persistent_exact_decode=False,
            persistent_padded_decode_steps=3,
            persistent_padded_decode_cuda_graph=True,
        )
        runner = FakePersistentPaddedBatchRunner()
        engine.runner = runner  # type: ignore[assignment]

        req = Request(prompt_token_ids=[1, 2, 3], max_new_tokens=4, req_id="solo")
        engine.add_request(req)

        engine.step()
        self.assertEqual(req.output_token_ids, [101])

        engine.step()
        self.assertEqual(runner.persistent_padded_starts, [([101], [3], 3)])
        self.assertEqual(runner.persistent_padded_reuses, [])
        self.assertEqual(req.output_token_ids, [101, 500])

        engine.step()
        self.assertEqual(runner.persistent_padded_reuses, [([500], [4])])
        self.assertEqual(req.output_token_ids, [101, 500, 610])

        finished = engine.step()
        self.assertEqual([req.req_id for req in finished], ["solo"])
        self.assertEqual(
            runner.persistent_padded_reuses,
            [([500], [4]), ([610], [5])],
        )
        self.assertEqual(runner.decode_batch_calls, [])
        self.assertEqual(runner.decode_step_calls, 0)
        self.assertEqual(engine.metrics.decode_model_calls, 3)
        self.assertEqual(engine.metrics.batched_decode_model_calls, 0)
        self.assertEqual(engine.metrics.fallback_decode_model_calls, 0)
        self.assertEqual(engine.metrics.padded_decode_batch_rows, 3)
        self.assertEqual(engine.metrics.persistent_padded_decode_rows, 3)
        self.assertEqual(engine.metrics.persistent_padded_decode_starts, 1)
        self.assertEqual(engine.metrics.persistent_padded_decode_reuses, 2)
        self.assertEqual(engine.metrics.persistent_padded_decode_splits, 1)
        self.assertFalse(engine._persistent_padded_decode_groups)
        self.assertEqual(runner.merged_cache.remaining_capacity_calls, 2)
        self.assertEqual(runner.merged_cache.commit_count, 1)
        self.assertTrue(req.status.is_finished)

    def test_decode_batch_respects_microbatch_row_cap(self) -> None:
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=2,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention", "sliding_attention"),
        )
        engine = GemmaNativeEngine(
            model=FakeModel(),
            config=cfg,
            num_slots=4,
            scheduler_config=SchedulerConfig(
                max_tokens_per_step=32,
                max_running_requests=4,
                max_tokens_per_request_per_step=8,
            ),
            decode_microbatch_rows=2,
        )
        runner = FakeBatchRunner()
        engine.runner = runner  # type: ignore[assignment]

        reqs = [
            Request(prompt_token_ids=[1, 2, 3], max_new_tokens=3, req_id=f"r{i}")
            for i in range(4)
        ]
        for req in reqs:
            engine.add_request(req)

        engine.step()
        engine.step()

        self.assertEqual(len(runner.decode_batch_calls), 2)
        self.assertEqual(runner.decode_batch_calls[0], ([101, 102], [3, 3]))
        self.assertEqual(runner.decode_batch_calls[1], ([103, 104], [3, 3]))
        self.assertEqual(runner.decode_step_calls, 0)
        self.assertEqual(engine.metrics.max_decode_batch_rows, 4)
        self.assertEqual(engine.metrics.max_decode_model_batch_rows, 2)
        self.assertEqual(engine.metrics.decode_microbatch_splits, 1)
        self.assertEqual(engine.metrics.decode_model_calls, 2)
        self.assertEqual(engine.metrics.batched_decode_model_calls, 2)
        self.assertEqual(engine.metrics.exact_decode_batch_rows, 4)

    def test_padded_decode_temp_metrics_are_recorded(self) -> None:
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=2,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention", "sliding_attention"),
        )
        engine = GemmaNativeEngine(
            model=FakeModel(),
            config=cfg,
            num_slots=2,
            scheduler_config=SchedulerConfig(
                max_tokens_per_step=16,
                max_running_requests=2,
                max_tokens_per_request_per_step=8,
            ),
        )
        runner = FakePaddedBatchRunner()
        engine.runner = runner  # type: ignore[assignment]

        reqs = [
            Request(prompt_token_ids=[1, 2, 3], max_new_tokens=3, req_id="a"),
            Request(prompt_token_ids=[4, 5, 6], max_new_tokens=3, req_id="b"),
        ]
        for req in reqs:
            engine.add_request(req)

        engine.step()
        engine.step()

        self.assertEqual(engine.metrics.padded_decode_batch_rows, 2)
        self.assertEqual(engine.metrics.exact_decode_batch_rows, 0)
        self.assertEqual(engine.metrics.padded_decode_temp_bytes, 3000)
        self.assertEqual(engine.metrics.padded_decode_temp_mask_bytes, 30)
        self.assertEqual(engine.metrics.padded_decode_copied_kv_bytes, 2100)
        self.assertEqual(engine.metrics.padded_decode_pad_kv_bytes, 600)
        self.assertEqual(engine.metrics.padded_decode_source_pad_kv_bytes, 450)
        self.assertEqual(engine.metrics.padded_decode_workspace_extra_pad_kv_bytes, 150)
        self.assertEqual(engine.metrics.padded_decode_reserved_kv_bytes, 300)
        self.assertEqual(engine.metrics.padded_decode_workspace_allocations, 1)
        self.assertEqual(engine.metrics.padded_decode_workspace_reuses, 1)
        self.assertEqual(engine.metrics.padded_decode_workspace_bypasses, 1)
        self.assertEqual(engine.metrics.max_padded_decode_temp_bytes, 3000)
        self.assertEqual(engine.metrics.max_padded_decode_pad_slots, 12)
        self.assertEqual(engine.metrics.max_padded_decode_workspace_extra_pad_slots, 5)
        self.assertAlmostEqual(engine.metrics.decode_timing_merge_s, 0.1)
        self.assertAlmostEqual(engine.metrics.decode_timing_model_forward_s, 0.2)
        self.assertAlmostEqual(engine.metrics.decode_timing_commit_s, 0.03)
        self.assertAlmostEqual(engine.metrics.decode_timing_split_s, 0.0)
        self.assertAlmostEqual(engine.metrics.decode_timing_mask_s, 0.04)
        self.assertAlmostEqual(engine.metrics.decode_timing_total_s, 0.37)
        exported = engine.metrics.as_dict()
        self.assertEqual(exported["padded_decode_temp_bytes"], 3000)
        self.assertEqual(exported["max_padded_decode_pad_slots"], 12)
        self.assertEqual(exported["padded_decode_workspace_reuses"], 1)
        self.assertEqual(exported["padded_decode_workspace_bypasses"], 1)
        self.assertEqual(exported["decode_timing_total_s"], 0.37)

    def test_decode_batch_respects_microbatch_byte_cap(self) -> None:
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=2,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention", "sliding_attention"),
        )
        engine = GemmaNativeEngine(
            model=FakeModel(),
            config=cfg,
            num_slots=4,
            scheduler_config=SchedulerConfig(
                max_tokens_per_step=32,
                max_running_requests=4,
                max_tokens_per_request_per_step=8,
            ),
            decode_microbatch_rows=0,
            decode_microbatch_bytes=1500,
        )
        runner = FakeBatchRunner()
        engine.runner = runner  # type: ignore[assignment]

        reqs = [
            Request(prompt_token_ids=[1, 2, 3], max_new_tokens=3, req_id=f"b{i}")
            for i in range(4)
        ]
        for req in reqs:
            engine.add_request(req)
            engine._caches[req.req_id] = FakeSizedCache([10, 10])  # type: ignore[assignment]
            engine._token_pool_commit_prefill_tokens(req, req.num_prompt_tokens)
            req.output_token_ids.append(101 + len(req.output_token_ids))
            req.num_computed_tokens = 3

        sampled = engine._execute_decode_batch(reqs)

        self.assertEqual(set(sampled), {req.req_id for req in reqs})
        self.assertEqual(len(runner.decode_batch_calls), 2)
        self.assertEqual(runner.decode_step_calls, 0)
        self.assertEqual(engine.metrics.max_decode_batch_rows, 4)
        self.assertEqual(engine.metrics.max_decode_model_batch_rows, 2)
        self.assertLessEqual(engine.metrics.max_decode_model_batch_bytes, 1500)
        self.assertEqual(engine.metrics.decode_microbatch_splits, 1)
        self.assertEqual(engine.metrics.decode_microbatch_byte_splits, 1)

    def test_length_bucketed_planner_groups_similar_cache_widths(self) -> None:
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
        )
        scheduler_engine = GemmaNativeEngine(
            model=FakeModel(),
            config=cfg,
            num_slots=4,
            scheduler_config=SchedulerConfig(
                max_tokens_per_step=32,
                max_running_requests=4,
                max_tokens_per_request_per_step=8,
            ),
            decode_microbatch_rows=2,
        )
        bucketed_engine = GemmaNativeEngine(
            model=FakeModel(),
            config=cfg,
            num_slots=4,
            scheduler_config=SchedulerConfig(
                max_tokens_per_step=32,
                max_running_requests=4,
                max_tokens_per_request_per_step=8,
            ),
            decode_microbatch_rows=2,
            decode_batch_planner="length_bucketed",
        )

        reqs = [
            Request(prompt_token_ids=[1], max_new_tokens=2, req_id="short-a"),
            Request(prompt_token_ids=[1], max_new_tokens=2, req_id="long-a"),
            Request(prompt_token_ids=[1], max_new_tokens=2, req_id="short-b"),
            Request(prompt_token_ids=[1], max_new_tokens=2, req_id="long-b"),
        ]
        lengths_by_req = {
            "short-a": [10],
            "long-a": [35],
            "short-b": [10],
            "long-b": [35],
        }
        for engine in (scheduler_engine, bucketed_engine):
            engine.runner = FakeBatchRunner()  # type: ignore[assignment]
            for req in reqs:
                engine.add_request(req)
                engine._caches[req.req_id] = FakeSizedCache(lengths_by_req[req.req_id])  # type: ignore[assignment]
                req.output_token_ids = [111]
                req.num_computed_tokens = 1

        scheduler_batches, scheduler_byte_split = scheduler_engine._plan_decode_model_batches(reqs)
        bucketed_batches, bucketed_byte_split = bucketed_engine._plan_decode_model_batches(reqs)

        self.assertFalse(scheduler_byte_split)
        self.assertFalse(bucketed_byte_split)
        self.assertEqual(
            [[req.req_id for req in batch] for batch in scheduler_batches],
            [["short-a", "long-a"], ["short-b", "long-b"]],
        )
        self.assertEqual(
            [[req.req_id for req in batch] for batch in bucketed_batches],
            [["short-a", "short-b"], ["long-a", "long-b"]],
        )
        scheduler_bytes = sum(
            scheduler_engine._estimate_decode_model_batch_bytes(batch)
            for batch in scheduler_batches
        )
        bucketed_bytes = sum(
            bucketed_engine._estimate_decode_model_batch_bytes(batch)
            for batch in bucketed_batches
        )
        self.assertLess(bucketed_bytes, scheduler_bytes)
        self.assertEqual(bucketed_engine.metrics.decode_length_bucketed_batches, 1)

    def test_invalid_decode_batch_planner_is_rejected(self) -> None:
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
        )
        with self.assertRaises(ValueError):
            GemmaNativeEngine(
                model=FakeModel(),
                config=cfg,
                num_slots=1,
                decode_batch_planner="unknown",  # type: ignore[arg-type]
            )

    def test_decode_workspace_is_opt_in(self) -> None:
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
        )
        default_engine = GemmaNativeEngine(
            model=FakeModel(),
            config=cfg,
            num_slots=1,
        )
        self.assertIsNone(default_engine.runner.decode_workspace)

        workspace_engine = GemmaNativeEngine(
            model=FakeModel(),
            config=cfg,
            num_slots=1,
            decode_workspace_bytes=1024,
            decode_workspace_width_bucket=8,
        )
        self.assertIsNotNone(workspace_engine.runner.decode_workspace)
        self.assertEqual(workspace_engine.decode_workspace_bytes, 1024)
        self.assertEqual(workspace_engine.decode_workspace_width_bucket, 8)

    def test_step_failure_finishes_scheduled_requests_as_error(self) -> None:
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
        )
        engine = GemmaNativeEngine(
            model=FakeModel(),
            config=cfg,
            num_slots=1,
            enable_token_pool_metadata=True,
            scheduler_config=SchedulerConfig(
                max_tokens_per_step=8,
                max_running_requests=1,
                max_tokens_per_request_per_step=8,
            ),
        )
        engine.runner = FakeFailingPrefillRunner()  # type: ignore[assignment]
        req = Request(prompt_token_ids=[1, 2, 3], max_new_tokens=2, req_id="err")
        engine.add_request(req)

        with self.assertRaisesRegex(RuntimeError, "synthetic prefill failure"):
            engine.step()

        self.assertIs(req.status, RequestStatus.FINISHED_ERROR)
        self.assertFalse(engine.has_unfinished)
        self.assertEqual(engine.arena.num_free_slots(), 1)
        self.assertNotIn(req.req_id, engine._break_masks)
        self.assertNotIn(req.req_id, engine._caches)
        self.assertEqual(engine.metrics.error_count, 1)
        self.assertEqual(engine.metrics.finished_requests, 1)
        self.assertEqual(engine.stats()["token_pool"]["active_request_slots"], 0)
        self.assertEqual(engine.stats()["token_pool"]["allocated_token_slots"], 0)
        trace = engine.finished_traces[req.req_id]
        self.assertEqual(trace.finish_reason, "error")
        self.assertEqual(trace.error, "synthetic prefill failure")

    def test_continuation_prefill_failure_finishes_request_as_error(self) -> None:
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
        )
        engine = GemmaNativeEngine(
            model=FakeModel(),
            config=cfg,
            num_slots=1,
            enable_token_pool_metadata=True,
            scheduler_config=SchedulerConfig(
                max_tokens_per_step=2,
                max_running_requests=1,
                max_tokens_per_request_per_step=2,
            ),
        )
        runner = FakeFailingContinuationPrefillRunner()
        engine.runner = runner  # type: ignore[assignment]
        req = Request(prompt_token_ids=[1, 2, 3, 4], max_new_tokens=2, req_id="err2")
        engine.add_request(req)

        engine.step()
        self.assertEqual(req.num_computed_tokens, 2)
        self.assertEqual(req.output_token_ids, [])
        self.assertIn(req.req_id, engine._caches)
        self.assertEqual(engine.stats()["token_pool"]["allocated_token_slots"], 2)

        with self.assertRaisesRegex(RuntimeError, "synthetic continuation prefill failure"):
            engine.step()

        self.assertIs(req.status, RequestStatus.FINISHED_ERROR)
        self.assertFalse(engine.has_unfinished)
        self.assertEqual(engine.arena.num_free_slots(), 1)
        self.assertNotIn(req.req_id, engine._break_masks)
        self.assertNotIn(req.req_id, engine._caches)
        self.assertEqual(engine.metrics.error_count, 1)
        self.assertEqual(engine.metrics.finished_requests, 1)
        self.assertEqual(engine.stats()["token_pool"]["active_request_slots"], 0)
        self.assertEqual(engine.stats()["token_pool"]["allocated_token_slots"], 0)
        trace = engine.finished_traces[req.req_id]
        self.assertEqual(trace.finish_reason, "error")
        self.assertEqual(trace.error, "synthetic continuation prefill failure")

    def test_finished_trace_limit_evicts_oldest_trace(self) -> None:
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
        )
        engine = GemmaNativeEngine(
            model=FakeModel(),
            config=cfg,
            num_slots=3,
            scheduler_config=SchedulerConfig(
                max_tokens_per_step=16,
                max_running_requests=3,
                max_tokens_per_request_per_step=8,
            ),
            finished_trace_limit=2,
        )
        engine.runner = FakeBatchRunner()  # type: ignore[assignment]
        for i in range(3):
            engine.add_request(
                Request(prompt_token_ids=[1, 2], max_new_tokens=1, req_id=f"r{i}")
            )

        engine.step()

        self.assertEqual(set(engine.finished_traces), {"r1", "r2"})
        self.assertEqual(engine.metrics.finished_requests, 3)


if __name__ == "__main__":
    unittest.main()
