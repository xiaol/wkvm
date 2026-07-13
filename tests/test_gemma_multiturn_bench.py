import types
import unittest

from experiments.gemma_multiturn_bench import (
    _turn_prompts_and_deltas,
    build_workload,
    extract_sglang_cached_tokens,
    extract_vllm_cached_tokens,
    percentile,
    request_order_indices,
    restore_logical_order,
    summarize_run,
    summarize_turn,
    workload_fingerprints,
)


class TestGemmaMultiturnBench(unittest.TestCase):
    def test_request_order_policies_are_deterministic_permutations(self) -> None:
        self.assertEqual(request_order_indices(4, 0, "forward"), [0, 1, 2, 3])
        self.assertEqual(request_order_indices(4, 0, "alternating"), [0, 1, 2, 3])
        self.assertEqual(request_order_indices(4, 1, "alternating"), [3, 2, 1, 0])
        shuffled = request_order_indices(8, 2, "seeded-shuffle", seed=17)
        self.assertEqual(sorted(shuffled), list(range(8)))
        self.assertEqual(
            shuffled,
            request_order_indices(8, 2, "seeded-shuffle", seed=17),
        )
        self.assertEqual(
            restore_logical_order(["d", "c", "b", "a"], [3, 2, 1, 0]),
            ["a", "b", "c", "d"],
        )

    def test_workload_is_deterministic_distinct_and_exact_length(self) -> None:
        kwargs = {
            "sessions": 3,
            "turns": 2,
            "initial_context_tokens": 96,
            "turn_input_tokens": 5,
            "vocab_size": 128,
        }
        first = build_workload(**kwargs)
        second = build_workload(**kwargs)

        self.assertEqual(first, second)
        self.assertEqual([len(prompt) for prompt in first.initial_prompts], [96] * 3)
        self.assertTrue(all(prompt[0] == 2 for prompt in first.initial_prompts))
        self.assertEqual(len({tuple(prompt) for prompt in first.initial_prompts}), 3)
        self.assertEqual(len(first.turn_deltas), 1)
        for deltas in first.turn_deltas:
            self.assertEqual([len(delta) for delta in deltas], [5] * 3)
            self.assertEqual(len({tuple(delta) for delta in deltas}), 3)

        fingerprints = workload_fingerprints(first)
        self.assertEqual(
            fingerprints["initial_prompts"]["prompt_total_tokens"],
            288,
        )
        self.assertEqual(
            [row["prompt_total_tokens"] for row in fingerprints["turn_deltas"]],
            [15],
        )
        self.assertEqual(
            len(fingerprints["initial_prompts"]["prompt_token_ids_sha256"]),
            64,
        )

    def test_turn_zero_uses_exact_initial_prompt_before_continuation_deltas(self) -> None:
        workload = build_workload(
            sessions=2,
            turns=3,
            initial_context_tokens=12,
            turn_input_tokens=3,
            vocab_size=64,
        )
        histories = [list(prompt) for prompt in workload.initial_prompts]

        prompts, deltas = _turn_prompts_and_deltas(workload, histories, 0)
        self.assertEqual(prompts, workload.initial_prompts)
        self.assertEqual(deltas, [[], []])
        self.assertEqual(histories, workload.initial_prompts)

        prompts, deltas = _turn_prompts_and_deltas(workload, histories, 1)
        self.assertEqual(deltas, workload.turn_deltas[0])
        self.assertEqual([len(prompt) for prompt in prompts], [15, 15])

    def test_summarize_turn_reports_accounting_percentiles_and_fingerprints(self) -> None:
        row = summarize_turn(
            turn_index=0,
            session_ids=["session-0000", "session-0001"],
            prompts=[list(range(10)), list(range(10, 20))],
            deltas=[[91, 92], [93, 94]],
            outputs=[[101, 102, 103], [201, 202, 203]],
            expected_output_tokens=3,
            new_input_tokens=[10, 10],
            wall_s=2.0,
            ttft_s=[0.1, 0.3],
            e2e_s=[0.4, 0.8],
            cached_tokens=[0, 8],
            errors=[None, None],
        )

        self.assertEqual(row["success_count"], 2)
        self.assertEqual(row["error_count"], 0)
        self.assertEqual(row["output_tokens"], 6)
        self.assertEqual(row["successful_new_input_tokens"], 20)
        self.assertEqual(row["useful_new_tokens"], 26)
        self.assertEqual(row["output_tok_s"], 3.0)
        self.assertEqual(row["useful_new_token_tok_s"], 13.0)
        self.assertEqual(row["p50_ttft_s"], 0.2)
        self.assertEqual(row["p95_ttft_s"], 0.29)
        self.assertEqual(row["p50_e2e_latency_s"], 0.6)
        self.assertEqual(row["p95_e2e_latency_s"], 0.78)
        self.assertEqual(row["cached_tokens_total"], 8)
        self.assertEqual(row["p50_cached_tokens"], 4.0)
        self.assertEqual(row["p95_cached_tokens"], 7.6)
        self.assertEqual(len(row["prompt_token_ids_sha256"]), 64)
        self.assertEqual(len(row["delta_token_ids_sha256"]), 64)
        self.assertEqual(len(row["request_output_token_ids_sha256"]), 64)
        self.assertTrue(row["output_fingerprint_complete"])

    def test_summarize_turn_counts_only_successful_useful_tokens(self) -> None:
        row = summarize_turn(
            turn_index=1,
            session_ids=["a", "b"],
            prompts=[[1, 2, 3], [4, 5, 6]],
            deltas=[[2], [5]],
            outputs=[[7, 8], [9]],
            expected_output_tokens=2,
            new_input_tokens=[1, 1],
            wall_s=1.0,
        )

        self.assertEqual(row["success_count"], 1)
        self.assertEqual(row["error_count"], 1)
        self.assertEqual(row["output_tokens"], 2)
        self.assertEqual(row["successful_new_input_tokens"], 1)
        self.assertEqual(row["useful_new_token_tok_s"], 3.0)
        self.assertFalse(row["output_fingerprint_complete"])
        self.assertIn("expected 2 output tokens", row["errors"][0]["error"])

    def test_percentile_and_run_summary_use_all_available_request_latencies(self) -> None:
        self.assertEqual(percentile([1.0, 3.0], 0.50), 2.0)
        self.assertAlmostEqual(percentile([1.0, 3.0], 0.95), 2.9)
        turn = summarize_turn(
            turn_index=0,
            session_ids=["a", "b"],
            prompts=[[1], [2]],
            deltas=[[3], [4]],
            outputs=[[5], [6]],
            expected_output_tokens=1,
            new_input_tokens=[1, 1],
            wall_s=2.0,
            ttft_s=[0.1, 0.3],
            e2e_s=[0.5, 0.9],
            cached_tokens=[0, 4],
        )

        summary = summarize_run([turn], requested_turns=1)

        self.assertTrue(summary["all_turns_recorded"])
        self.assertEqual(summary["output_tok_s"], 1.0)
        self.assertEqual(summary["useful_new_token_tok_s"], 2.0)
        self.assertEqual(summary["p50_ttft_s"], 0.2)
        self.assertEqual(summary["p95_e2e_latency_s"], 0.88)
        self.assertEqual(summary["cached_tokens_total"], 4)
        self.assertEqual(summary["completed_requests_per_s"], 1.0)
        self.assertTrue(summary["cache_telemetry_complete"])
        self.assertEqual(summary["turn_0"]["output_tok_s"], 1.0)
        self.assertEqual(summary["continuation_turns"]["turn_rows"], 0)

    def test_run_summary_separates_turn_zero_and_continuations(self) -> None:
        rows = [
            summarize_turn(
                turn_index=turn_index,
                session_ids=["a", "b"],
                prompts=[[1], [2]],
                deltas=[[], []] if turn_index == 0 else [[3], [4]],
                outputs=[[5, 6], [7, 8]],
                expected_output_tokens=2,
                new_input_tokens=[1, 1],
                wall_s=2.0 if turn_index == 0 else 1.0,
                cached_tokens=[0, 0] if turn_index == 0 else [1, 1],
            )
            for turn_index in range(3)
        ]

        summary = summarize_run(rows, requested_turns=3)

        self.assertEqual(summary["output_tok_s"], 3.0)
        self.assertEqual(summary["turn_0"]["output_tok_s"], 2.0)
        self.assertEqual(summary["continuation_turns"]["output_tok_s"], 4.0)
        self.assertEqual(
            summary["continuation_turns"]["completed_requests_per_s"],
            2.0,
        )
        self.assertEqual(summary["cached_tokens_available_count"], 6)

    def test_cached_token_extractors_use_actual_incumbent_fields(self) -> None:
        vllm_outputs = [
            types.SimpleNamespace(num_cached_tokens=7, metrics=None),
            types.SimpleNamespace(
                num_cached_tokens=None,
                metrics=types.SimpleNamespace(num_cached_tokens=11),
            ),
            types.SimpleNamespace(
                num_cached_tokens=None,
                metrics=types.SimpleNamespace(cached_tokens=13),
            ),
            types.SimpleNamespace(num_cached_tokens=None, metrics=None),
        ]
        sglang_outputs = [
            {"meta_info": {"cached_tokens": 5}},
            {"meta_info": {"cached_tokens": 0}},
            {"meta_info": {}},
        ]

        self.assertEqual(
            extract_vllm_cached_tokens(vllm_outputs),
            [7, 11, 13, None],
        )
        self.assertEqual(
            extract_sglang_cached_tokens(sglang_outputs),
            [5, 0, None],
        )


if __name__ == "__main__":
    unittest.main()
