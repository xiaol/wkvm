from __future__ import annotations

import argparse
from types import SimpleNamespace
import unittest

from experiments.native_gemma_quality import (
    QualityCase,
    build_parser,
    build_quality_cases,
    new_routed_cache_observations,
    normalize_args,
    observe_routed_caches,
    parse_depth_csv,
    parse_int_csv,
    quality_validation,
    run_engine_cases,
    runtime_validation,
    scorer_metadata,
    summarize_scores,
)


class _Tokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


class _FakeEngine:
    def __init__(self, outputs, *, finish_status=None):
        self.outputs = outputs
        self.finish_status = finish_status
        self.requests = []
        self._unfinished = True
        self._caches = {}

    @property
    def has_unfinished(self):
        return self._unfinished

    def add_request(self, request, *, break_mask=None):
        self.requests.append((request, break_mask))

    def step(self):
        from wkvm.core.request import RequestStatus

        for request, _break_mask in self.requests:
            request.output_token_ids = list(self.outputs[request.req_id])
            request.status = self.finish_status or RequestStatus.FINISHED_LENGTH
        self._unfinished = False
        return [request for request, _break_mask in self.requests]


class TestNativeGemmaQuality(unittest.TestCase):
    def test_csv_parsers_and_winning_defaults(self) -> None:
        self.assertEqual(parse_int_csv("2048, 8192"), (2048, 8192))
        self.assertEqual(parse_depth_csv("0.1, .5,1"), (0.1, 0.5, 1.0))
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_int_csv("0")
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_depth_csv("1.1")

        args = build_parser().parse_args([])
        normalize_args(args)
        self.assertEqual(args.chunk, 2048)
        self.assertEqual(args.route_chunk, 512)
        self.assertEqual(args.prefill_microbatch_rows, 2)
        self.assertEqual(args.token_pool_capacity, 36_864)
        self.assertEqual(args.token_pool_max_context_len, 8448)

    def test_scorer_metadata_uses_explicit_then_closure(self) -> None:
        expected = "AZURE-123"
        closure_scorer = lambda output: float(expected in output)
        self.assertEqual(
            scorer_metadata(closure_scorer),
            {
                "source": "scorer.closure",
                "value": {"expected": "AZURE-123"},
            },
        )
        explicit_scorer = lambda output: float(bool(output))
        explicit_scorer.meta = {"target": "Aurora"}
        self.assertEqual(
            scorer_metadata(explicit_scorer),
            {"source": "scorer.meta", "value": {"target": "Aurora"}},
        )

    def test_build_cases_uses_legacy_seed_shape_and_exact_context(self) -> None:
        seen = []

        def task(_tokenizer, rng, context, depth):
            marker = rng.randint(1, 1_000_000)
            seen.append((context, depth, marker))
            expected = str(marker)
            scorer = lambda output: float(expected in output)
            return [[marker] * context], scorer, 3

        cases = build_quality_cases(
            _Tokenizer(),
            contexts=(4,),
            depths=(0.25, 0.75),
            t12_seeds=2,
            t3_seeds=1,
            tasks=(("t1-needle", task), ("t3-aggregate", task)),
            break_mask_builder=lambda _tokenizer, ids: [False] * len(ids),
        )

        self.assertEqual(len(cases), 6)
        self.assertEqual([case.seed for case in cases], [0, 1, 0, 1, 0, 0])
        self.assertTrue(all(len(case.prompt_token_ids) == 4 for case in cases))
        self.assertEqual(
            cases[0].seed_key,
            "4/t1-needle/0.25/0",
        )
        self.assertEqual(len(seen), 6)

    def test_engine_rows_score_pre_stop_text_and_fingerprint_tokens(self) -> None:
        cases = [
            QualityCase(
                req_id="quality-a",
                task="t1-needle",
                context_tokens=4,
                depth=0.5,
                seed=0,
                seed_key="4/t1-needle/0.5/0",
                prompt_token_ids=(1, 2, 3, 4),
                break_mask=(False, False, False, False),
                max_new_tokens=3,
                scorer=lambda output: float(output == "7"),
                scorer_metadata={},
            ),
            QualityCase(
                req_id="quality-b",
                task="t3-aggregate",
                context_tokens=4,
                depth=0.5,
                seed=0,
                seed_key="4/t3-aggregate/0.5/0",
                prompt_token_ids=(5, 6, 7, 8),
                break_mask=(False, False, False, False),
                max_new_tokens=3,
                scorer=lambda output: float(output == "8 9 10"),
                scorer_metadata={},
            ),
        ]
        engine = _FakeEngine(
            {
                "quality-a": [7, 106, 99],
                "quality-b": [8, 9, 10],
            }
        )

        rows, observations, requests = run_engine_cases(
            engine,
            _Tokenizer(),
            cases,
            max_steps=2,
        )

        self.assertEqual([row["score"] for row in rows], [1.0, 1.0])
        self.assertEqual(rows[0]["output_text"], "7")
        self.assertEqual(rows[0]["output_token_ids"], [7, 106, 99])
        self.assertEqual(
            rows[0]["generated_output_fingerprint"]["output_token_count"],
            3,
        )
        self.assertEqual(len(rows[0]["prompt_fingerprint"]["prompt_token_ids_sha256"]), 64)
        self.assertEqual(len(rows[0]["output_text_sha256"]), 64)
        self.assertEqual(observations["routed_layer_samples"], 0)
        self.assertEqual(len(requests), 2)
        summary = summarize_scores(rows)
        self.assertEqual(summary["overall_mean_score"], 1.0)
        self.assertEqual(summary["overall_cell_mean_score"], 1.0)
        self.assertEqual(summary["cell_count"], 2)
        self.assertEqual(summary["by_task"]["t3-aggregate"]["mean_score"], 1.0)

    def test_engine_error_status_cannot_score_as_success(self) -> None:
        from wkvm.core.request import RequestStatus

        case = QualityCase(
            req_id="quality-error",
            task="t1-needle",
            context_tokens=4,
            depth=0.5,
            seed=0,
            seed_key="4/t1-needle/0.5/0",
            prompt_token_ids=(1, 2, 3, 4),
            break_mask=(False, False, False, False),
            max_new_tokens=3,
            scorer=lambda output: 1.0,
            scorer_metadata={},
        )
        engine = _FakeEngine(
            {"quality-error": [7, 8, 9]},
            finish_status=RequestStatus.FINISHED_ERROR,
        )

        rows, _observations, _requests = run_engine_cases(
            engine,
            _Tokenizer(),
            [case],
            max_steps=2,
        )

        self.assertFalse(rows[0]["successful"])
        self.assertIsNone(rows[0]["score"])

    def test_quality_validation_is_separate_from_runtime_path_validation(self) -> None:
        rows = []
        for task in ("t1-needle", "t2-multikey", "t3-aggregate"):
            rows.append(
                {
                    "context_tokens": 8192,
                    "task": task,
                    "depth": 0.5,
                    "successful": True,
                    "score": 0.0,
                }
            )
        summary = summarize_scores(rows)

        result = quality_validation(
            rows=rows,
            summary=summary,
            workload={
                "contexts": [8192],
                "depths": [0.5],
                "t12_seeds": 1,
                "t3_seeds": 1,
                "case_count": 3,
            },
            require_full_grid=False,
        )

        self.assertFalse(result["passed"])
        self.assertIn("overall_cell_mean_below_gate", result["violations"])
        self.assertNotIn("full_quality_grid_not_observed", result["violations"])

    def test_quality_validation_checks_full_grid_contract(self) -> None:
        rows = []
        for context_tokens in (8192, 16384, 32768):
            for task in ("t1-needle", "t2-multikey", "t3-aggregate"):
                seed_count = 1 if task == "t3-aggregate" else 3
                score = 5.0 / 6.0 if task == "t3-aggregate" else 1.0
                for depth in (0.1, 0.3, 0.5, 0.7, 0.9):
                    for _seed in range(seed_count):
                        rows.append(
                            {
                                "context_tokens": context_tokens,
                                "task": task,
                                "depth": depth,
                                "successful": True,
                                "score": score,
                            }
                        )
        summary = summarize_scores(rows)
        result = quality_validation(
            rows=rows,
            summary=summary,
            workload={
                "contexts": [8192, 16384, 32768],
                "depths": [0.1, 0.3, 0.5, 0.7, 0.9],
                "t12_seeds": 3,
                "t3_seeds": 1,
                "case_count": 105,
            },
            require_full_grid=True,
        )

        self.assertTrue(result["passed"])
        self.assertTrue(result["full_grid_observed"])

    def test_cache_observation_records_folding_and_bounds(self) -> None:
        layer = SimpleNamespace(
            _evicted=1024,
            _pend_k=SimpleNamespace(shape=(1, 2, 127, 4)),
            _n_active=11,
            _dense_storage_released=True,
            cumulative_length=4096,
            materialized_tokens=lambda: 1536,
        )
        engine = SimpleNamespace(
            _caches={"request": SimpleNamespace(layers=[layer])}
        )
        observations = new_routed_cache_observations()

        observe_routed_caches(engine, observations)

        self.assertEqual(observations["max_evicted_tokens"], 1024)
        self.assertEqual(observations["max_pending_tokens"], 127)
        self.assertEqual(observations["max_active_route_slots"], 11)
        self.assertEqual(observations["minimum_materialized_fraction"], 0.375)
        self.assertTrue(observations["dense_storage_release_observed"])

    def test_runtime_validation_proves_native_fold_and_microbatch(self) -> None:
        config = {
            "chunk": 2048,
            "route_chunk": 512,
            "sink": 16,
            "window": 1024,
            "slots": 2,
            "prefill_microbatch_rows": 2,
        }
        stats = {
            "uses_hf_transformer_forward": False,
            "uses_hf_model_construction": False,
            "native_gemma_checkpoint_loader": True,
            "token_pool_attention_enabled": True,
            "max_prefill_batch_rows": 2,
        }
        rows = [
            {"context_tokens": 8192, "successful": True, "score": 1.0},
            {"context_tokens": 8192, "successful": True, "score": 0.5},
        ]
        observations = {
            "routed_layer_samples": 4,
            "max_evicted_tokens": 4096,
            "max_pending_tokens": 511,
        }
        triton = {
            "effective_enabled": True,
            "runtime_errors": 0,
            "fallback_reasons": {},
        }

        result = runtime_validation(
            config=config,
            engine_stats=stats,
            rows=rows,
            observations=observations,
            triton_stats=triton,
        )

        self.assertTrue(result["passed"])
        self.assertTrue(result["fold_observed"])
        self.assertTrue(result["batched_prefill_observed"])

        observations["max_evicted_tokens"] = 0
        failed = runtime_validation(
            config=config,
            engine_stats=stats,
            rows=rows,
            observations=observations,
            triton_stats=triton,
        )
        self.assertFalse(failed["passed"])
        self.assertIn("routed_fold_not_observed", failed["violations"])

    def test_runtime_validation_requires_exact_b16_winning_shape(self) -> None:
        config = {
            "chunk": 2048,
            "route_chunk": 512,
            "sink": 16,
            "window": 1024,
            "m_slots": 64,
            "slots": 16,
            "prefill_microbatch_rows": 2,
            "decode_microbatch_rows": 16,
            "persistent_padded_decode_steps": 128,
            "persistent_padded_decode_graph_warmup_iters": 0,
            "persistent_padded_sliding_metadata_padding": True,
            "native_gemma_checkpoint_loader": True,
            "use_native_gemma_forward": True,
            "native_gemma_attention_backend": "sdpa_single_gqa",
            "native_gemma_projection_backend": "separate",
            "enable_token_pool_attention": True,
            "token_pool_capacity": 36_864,
            "token_pool_max_context_len": 33_024,
            "token_pool_paged_block_size": 16,
        }
        row = {
            "req_id": "quality-a",
            "context_tokens": 32768,
            "successful": True,
            "score": 1.0,
        }
        stats = {
            "uses_hf_transformer_forward": False,
            "uses_hf_model_construction": False,
            "native_gemma_checkpoint_loader": True,
            "token_pool_attention_enabled": True,
            "error_count": 0,
            "max_prefill_batch_rows": 2,
            "max_decode_batch_rows": 16,
            "persistent_padded_decode_cuda_graph_captures": 1,
            "persistent_padded_decode_cuda_graph_replays": 1,
            "requests": {
                "quality-a": {
                    "error": None,
                    "finish_reason": "length",
                    "output_tokens": 32,
                    "target_output_tokens": 32,
                }
            },
        }
        observations = {
            "routed_layer_samples": 1,
            "max_evicted_tokens": 1,
            "max_pending_tokens": 1,
            "dense_storage_release_observed": True,
        }
        triton = {
            "effective_enabled": True,
            "runtime_errors": 0,
            "fallback_reasons": {},
            "successes": 1,
            "paged_successes": 1,
        }

        result = runtime_validation(
            config=config,
            engine_stats=stats,
            rows=[row],
            observations=observations,
            triton_stats=triton,
            require_benchmark_shape=True,
        )

        self.assertTrue(result["passed"])
        self.assertTrue(result["benchmark_shape_observed"])
        config["m_slots"] = 32
        failed = runtime_validation(
            config=config,
            engine_stats=stats,
            rows=[row],
            observations=observations,
            triton_stats=triton,
            require_benchmark_shape=True,
        )
        self.assertFalse(failed["passed"])
        self.assertIn("benchmark_b16_winning_shape_not_observed", failed["violations"])


if __name__ == "__main__":
    unittest.main()
