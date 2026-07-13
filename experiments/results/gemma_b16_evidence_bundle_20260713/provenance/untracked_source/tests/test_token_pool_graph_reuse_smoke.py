import unittest

from experiments.token_pool_graph_reuse_smoke import (
    MAX_BATCH_SIZE,
    MAX_GRAPH_WARMUP_ITERS,
    MAX_HIT_COHORTS,
    MAX_ROW_WIDTH,
    CohortMeasurement,
    SmokeConfig,
    SmokeInvariantError,
    build_cohort_plans,
    expected_signals,
    validate_and_summarize,
)


class TestTokenPoolGraphReuseSmoke(unittest.TestCase):
    def test_cohort_plans_are_bounded_deterministic_and_disjoint(self) -> None:
        config = SmokeConfig(
            batch_size=2,
            row_width=3,
            hit_cohorts=2,
            graph_warmup_iters=0,
        )

        plans = build_cohort_plans(config)

        self.assertEqual(len(plans), 3)
        self.assertEqual(plans, build_cohort_plans(config))
        self.assertEqual(plans[0].token_ids, plans[1].token_ids)
        self.assertEqual(plans[0].position_ids, plans[1].position_ids)
        all_slots = [
            slot
            for plan in plans
            for row in plan.token_slot_rows
            for slot in row
        ]
        self.assertEqual(len(all_slots), len(set(all_slots)))
        self.assertEqual(plans[0].out_cache_loc, (2, 5))
        self.assertEqual(plans[1].out_cache_loc, (8, 11))
        self.assertNotEqual(expected_signals(plans[0]), expected_signals(plans[1]))

    def test_config_enforces_hard_runtime_bounds(self) -> None:
        invalid = (
            SmokeConfig(batch_size=0),
            SmokeConfig(batch_size=MAX_BATCH_SIZE + 1),
            SmokeConfig(row_width=0),
            SmokeConfig(row_width=MAX_ROW_WIDTH + 1),
            SmokeConfig(hit_cohorts=0),
            SmokeConfig(hit_cohorts=MAX_HIT_COHORTS + 1),
            SmokeConfig(graph_warmup_iters=-1),
            SmokeConfig(graph_warmup_iters=MAX_GRAPH_WARMUP_ITERS + 1),
            SmokeConfig(device="cpu"),
        )
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                config.validate()

    def test_summary_proves_one_capture_hits_and_metadata_refresh(self) -> None:
        plans = build_cohort_plans(
            SmokeConfig(batch_size=2, row_width=2, hit_cohorts=2)
        )
        measurements = tuple(
            CohortMeasurement(
                cohort_index=plan.cohort_index,
                captured=int(plan.cohort_index == 0),
                cache_hit=int(plan.cohort_index > 0),
                synchronized_wall_s=0.010 - plan.cohort_index * 0.002,
                runner_graph_prepare_wall_s=(
                    0.007 if plan.cohort_index == 0 else 0.0001
                ),
                runner_decode_wall_s=0.0002,
                runner_replay_dispatch_wall_s=0.0001,
                runner_metadata_copy_wall_s=0.00005,
                metadata_tensor_copies=int(plan.cohort_index > 0) * 6,
                metadata_tensor_copy_skips=1,
                actual_signals=expected_signals(plan),
            )
            for plan in plans
        )

        summary = validate_and_summarize(
            plans,
            measurements,
            graph_cache_entries=1,
        )

        self.assertEqual(summary["proof"]["capture_count"], 1)
        self.assertEqual(summary["proof"]["cache_hit_count"], 2)
        self.assertEqual(summary["proof"]["hit_metadata_tensor_copies"], 12)
        self.assertAlmostEqual(
            summary["timing_ms"]["hit_cohort_synchronized_wall"]["median"],
            7.0,
        )

    def test_summary_rejects_stale_or_uncopied_hit_metadata(self) -> None:
        plans = build_cohort_plans(
            SmokeConfig(batch_size=1, row_width=2, hit_cohorts=1)
        )
        measurements = (
            CohortMeasurement(
                cohort_index=0,
                captured=1,
                cache_hit=0,
                synchronized_wall_s=0.01,
                runner_graph_prepare_wall_s=0.008,
                runner_decode_wall_s=0.001,
                runner_replay_dispatch_wall_s=0.0005,
                runner_metadata_copy_wall_s=0.0001,
                metadata_tensor_copies=0,
                metadata_tensor_copy_skips=1,
                actual_signals=expected_signals(plans[0]),
            ),
            CohortMeasurement(
                cohort_index=1,
                captured=0,
                cache_hit=1,
                synchronized_wall_s=0.002,
                runner_graph_prepare_wall_s=0.0001,
                runner_decode_wall_s=0.001,
                runner_replay_dispatch_wall_s=0.0005,
                runner_metadata_copy_wall_s=0.0001,
                metadata_tensor_copies=0,
                metadata_tensor_copy_skips=0,
                actual_signals=expected_signals(plans[0]),
            ),
        )

        with self.assertRaisesRegex(
            SmokeInvariantError,
            "output did not reflect its metadata.*did not copy fresh cohort metadata",
        ):
            validate_and_summarize(
                plans,
                measurements,
                graph_cache_entries=1,
            )


if __name__ == "__main__":
    unittest.main()
