import unittest
from types import SimpleNamespace

from wkvm.core.arena import StateArena
from wkvm.models.gemma import gemma4_e4b_routed_span_config
from wkvm.runner.gemma_state import (
    GemmaRoutedStateBank,
    RoutedSpanLayerState,
    SpanRecord,
    pad_valid_masks,
)


class TestGemmaRoutedCache(unittest.TestCase):
    def _config(self):
        return gemma4_e4b_routed_span_config(
            ring_tokens=8,
            pending_tokens=4,
            routed_slots=4,
            span_budget_tokens=6,
            max_span_tokens=3,
        )

    def test_state_spec_allocates_gemma_families(self) -> None:
        cfg = self._config()
        spec = cfg.state_spec()
        names = {family.name for family in spec.families}
        self.assertIn("gemma_sliding_kv", names)
        self.assertIn("gemma_routed_span", names)
        self.assertIn("gemma_routed_meta", names)
        arena = StateArena(spec, num_slots=2)
        slots = arena.allocate()
        self.assertIn("gemma_routed_span", slots)
        self.assertNotEqual(slots["gemma_routed_span"], 0)

    def test_routed_materialized_bound_matches_runtime_state(self) -> None:
        cfg = gemma4_e4b_routed_span_config()
        expected = (
            cfg.sink_tokens
            + cfg.ring_tokens
            + cfg.pending_tokens
            - 1
            + cfg.routed_slots
            * (1 + max(cfg.span_budget_tokens, cfg.max_span_tokens))
        )
        self.assertEqual(cfg.routed_materialized_tokens, expected)
        self.assertEqual(cfg.routed_materialized_tokens, 10_831)

    def test_state_spec_accounts_for_all_persistent_routed_tensors(self) -> None:
        cfg = gemma4_e4b_routed_span_config()

        self.assertEqual(cfg.routed_authoritative_tokens, 10_767)
        self.assertEqual(cfg.routed_readout_layer_bytes, 44_363_776)
        self.assertEqual(cfg.routed_authoritative_layer_bytes, 44_101_632)
        self.assertEqual(cfg.routed_slot_summary_layer_bytes, 524_288)
        self.assertEqual(cfg.routed_leader_state_bytes, 266_240)
        routed_family = next(
            family
            for family in cfg.state_spec().families
            if family.name == "gemma_routed_span"
        )
        self.assertEqual(routed_family.bytes_per_slot, 356_225_024)
        self.assertEqual(cfg.state_spec().bytes_per_request, 440_168_764)

    def test_explicit_no_break_mask_keeps_pending_state_bounded(self) -> None:
        cfg = self._config()
        bank = GemmaRoutedStateBank(cfg, num_slots=1)
        slots = {"gemma_routed_span": 1}
        positions = list(range(100))
        bank.ingest_positions(slots, positions, [False] * len(positions))

        for layer in bank.slot_state(slots).full_layers.values():
            self.assertLess(len(layer.pending_positions), cfg.pending_tokens)
            self.assertLessEqual(
                len(layer.materialized_positions()),
                cfg.routed_materialized_tokens,
            )

    def test_metadata_no_break_spans_preserve_native_fallback_semantics(self) -> None:
        positions = list(range(48))
        for max_span_tokens, expected_widths in ((48, [24, 24]), (8, [8] * 6)):
            with self.subTest(max_span_tokens=max_span_tokens):
                cfg = gemma4_e4b_routed_span_config(
                    num_hidden_layers=1,
                    num_kv_shared_layers=0,
                    layer_types=("full_attention",),
                    pending_tokens=48,
                    routed_slots=1,
                    span_budget_tokens=1,
                    max_span_tokens=max_span_tokens,
                )
                layer = RoutedSpanLayerState(0, cfg)
                spans = layer._split_spans(positions, [False] * len(positions))

                self.assertEqual([len(span) for span in spans], expected_widths)
                layer.add_span(tuple(spans[0]), 0)
                self.assertEqual(layer.bank_occupancy_tokens, expected_widths[0])
                self.assertEqual(
                    len(layer.materialized_positions()),
                    expected_widths[0] + 1,
                )

    def test_native_no_break_fallback_respects_configured_span_max(self) -> None:
        from wkvm.runner.gemma_runner import NativeRoutedSpanLayer

        for max_span_tokens, expected_spans in (
            (48, [(0, 24), (24, 48)]),
            (8, [(0, 8), (8, 16), (16, 24), (24, 32), (32, 40), (40, 48)]),
        ):
            with self.subTest(max_span_tokens=max_span_tokens):
                cfg = gemma4_e4b_routed_span_config(
                    num_hidden_layers=1,
                    num_kv_shared_layers=0,
                    layer_types=("full_attention",),
                    pending_tokens=48,
                    routed_slots=1,
                    span_budget_tokens=1,
                    max_span_tokens=max_span_tokens,
                )
                layer = NativeRoutedSpanLayer(0, cfg)
                layer.break_mask = [False] * 48

                spans, routed_tokens = layer._split_spans(0, 48)

                self.assertEqual(spans, expected_spans)
                self.assertEqual(routed_tokens, 48)

    def test_native_routed_width_respects_bound_below_fallback_span(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("full_attention",),
            num_kv_heads=1,
            head_dim=2,
            sink_tokens=1,
            ring_tokens=1,
            pending_tokens=24,
            routed_slots=1,
            reps_per_slot=1,
            span_budget_tokens=1,
            max_span_tokens=1,
            sliding_window=1,
        )
        hf_cfg = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("full_attention",),
            sliding_window=1,
        )
        cache = NativeGemmaRoutedCache(hf_cfg, cfg)
        cache.set_span_break_mask([False] * 64)
        first_width = cfg.sink_tokens + cfg.ring_tokens + cfg.pending_tokens
        first_keys = torch.arange(
            first_width * cfg.head_dim,
            dtype=torch.float32,
        ).reshape(1, 1, first_width, cfg.head_dim)
        cache.update(first_keys, first_keys + 100, layer_idx=0)
        layer = cache.layers[0]

        self.assertEqual(
            [int(span["k"].shape[2]) for span in layer._slot_spans[0]],
            [1],
        )
        self.assertEqual(layer._bank_span_tokens, 1)
        self.assertEqual(layer._active_span_slots, 1)
        self.assertEqual(
            layer.span_storage_bytes(),
            sum(
                tensor.numel() * tensor.element_size()
                for span in layer._slot_spans[0]
                for tensor in (span["k"], span["v"])
            ),
        )
        self.assertEqual(int(layer._pend_k.shape[2]), 0)

        pending_width = cfg.pending_tokens - 1
        pending_keys = torch.arange(
            pending_width * cfg.head_dim,
            dtype=torch.float32,
        ).reshape(1, 1, pending_width, cfg.head_dim)
        cache.update(pending_keys, pending_keys + 200, layer_idx=0)

        self.assertEqual(int(layer._pend_k.shape[2]), pending_width)
        self.assertEqual(
            layer.materialized_tokens(),
            cfg.routed_materialized_tokens,
        )
        self.assertLessEqual(
            layer.materialized_tokens(),
            cfg.routed_materialized_tokens,
        )

    def test_routed_decision_coordination_compacts_after_followers(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=4,
            num_kv_shared_layers=0,
            layer_types=("full_attention",) * 4,
            num_kv_heads=1,
            head_dim=2,
            sink_tokens=1,
            ring_tokens=1,
            pending_tokens=2,
            routed_slots=2,
            reps_per_slot=1,
            span_budget_tokens=2,
            max_span_tokens=1,
            sliding_window=1,
        )
        hf_cfg = SimpleNamespace(
            num_hidden_layers=4,
            num_kv_shared_layers=0,
            layer_types=("full_attention",) * 4,
            sliding_window=1,
        )
        cache = NativeGemmaRoutedCache(hf_cfg, cfg)
        cache.set_span_break_mask([False] * 16)
        coordinator = cache.layers[0].coord
        self.assertIsNotNone(coordinator)

        for offset in (0, 8):
            keys = torch.arange(8, dtype=torch.float32).reshape(1, 1, 4, 2)
            keys = keys + offset
            for layer_idx in range(4):
                cache.update(keys, keys + 100, layer_idx=layer_idx)
            self.assertEqual(coordinator.pending_operations, 0)

    def test_state_bytes_includes_authoritative_routed_storage(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("full_attention",),
            num_kv_heads=1,
            head_dim=2,
            sink_tokens=1,
            ring_tokens=2,
            pending_tokens=8,
            routed_slots=2,
            reps_per_slot=1,
            span_budget_tokens=2,
            max_span_tokens=2,
            sliding_window=2,
        )
        hf_cfg = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("full_attention",),
            sliding_window=2,
        )
        cache = NativeGemmaRoutedCache(hf_cfg, cfg)
        keys = torch.arange(12, dtype=torch.float32).reshape(1, 1, 3, 4)[..., :2]
        cache.update(keys, keys + 100, layer_idx=0)
        layer = cache.layers[0]
        dense_bytes = sum(
            tensor.numel() * tensor.element_size()
            for tensor in (layer.keys, layer.values)
        )

        self.assertGreater(cache.state_bytes(), dense_bytes)
        before_release = cache.state_bytes()
        self.assertTrue(layer.release_dense_materialized_storage())
        self.assertLess(cache.state_bytes(), before_release)
        self.assertGreater(cache.state_bytes(), 0)

    def test_ring_capacity_is_constant_after_long_prefill(self) -> None:
        cfg = self._config()
        bank = GemmaRoutedStateBank(cfg, num_slots=1)
        slots = {"gemma_routed_span": 1}
        bank.zero_slots(slots)
        breaks = [False] * 100
        for i in range(2, 100, 5):
            breaks[i] = True
        bank.ingest_positions(slots, list(range(100)), breaks)
        state = bank.slot_state(slots)
        for layer in state.full_layers.values():
            self.assertLessEqual(len(layer.ring_positions), cfg.ring_tokens)
            self.assertEqual(len(layer.sink_positions), cfg.sink_tokens)

    def test_sink_tokens_remain_stable(self) -> None:
        cfg = self._config()
        bank = GemmaRoutedStateBank(cfg, num_slots=1)
        slots = {"gemma_routed_span": 1}
        bank.ingest_positions(slots, list(range(20)))
        first = {
            layer_id: tuple(layer.sink_positions)
            for layer_id, layer in bank.slot_state(slots).full_layers.items()
        }
        bank.ingest_positions(slots, list(range(20, 80)))
        second = {
            layer_id: tuple(layer.sink_positions)
            for layer_id, layer in bank.slot_state(slots).full_layers.items()
        }
        self.assertEqual(first, second)
        self.assertTrue(all(v == tuple(range(cfg.sink_tokens)) for v in second.values()))

    def test_span_routing_is_value_based_not_rope_key_based(self) -> None:
        with self.assertRaises(ValueError):
            SpanRecord(positions=(1, 2, 3), route_slot=0, feature_kind="rope_key")
        span = SpanRecord(positions=(1, 2, 3), route_slot=0, feature_kind="value")
        self.assertEqual(span.feature_kind, "value")

    def test_padded_valid_masks_hide_pad_slots(self) -> None:
        cfg = self._config()
        bank = GemmaRoutedStateBank(cfg, num_slots=2)
        a = {"gemma_routed_span": 1}
        b = {"gemma_routed_span": 2}
        bank.ingest_positions(a, list(range(30)))
        bank.ingest_positions(b, list(range(50)))
        masks = pad_valid_masks([bank.slot_state(a), bank.slot_state(b)])
        self.assertTrue(masks)
        for rows in masks.values():
            self.assertEqual(len(rows), 2)
            self.assertEqual(len(rows[0]), len(rows[1]))
            self.assertIn(False, rows[0] + rows[1])
            for row in rows:
                if False in row:
                    first_pad = row.index(False)
                    self.assertTrue(all(not v for v in row[first_pad:]))

    def test_memory_estimate_is_monotonic_and_context_bounded(self) -> None:
        cfg = self._config()
        bank = GemmaRoutedStateBank(cfg, num_slots=4)
        one = bank.memory_accounting(resident_sessions=1)
        two = bank.memory_accounting(resident_sessions=2)
        self.assertGreater(two["estimated_bytes"], one["estimated_bytes"])
        slots = {"gemma_routed_span": 1}
        bank.ingest_positions(slots, list(range(32)))
        short = bank.slot_state(slots).resident_tokens
        bank.ingest_positions(slots, list(range(32, 512)))
        long = bank.slot_state(slots).resident_tokens
        max_per_layer = cfg.routed_materialized_tokens
        self.assertLessEqual(long, max_per_layer * len(cfg.full_kv_layers))
        self.assertGreaterEqual(long, short)

    def test_exact_decode_merge_splits_back_to_original_caches(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        hf_cfg = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=16,
        )
        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=16,
        )
        caches = [NativeGemmaRoutedCache(hf_cfg, cfg) for _ in range(2)]
        for row, cache in enumerate(caches):
            key = torch.full((1, 1, 3, 2), float(row + 1))
            value = torch.full((1, 1, 3, 2), float(row + 11))
            cache.update(key, value, layer_idx=0)

        merged, info = NativeGemmaRoutedCache.merge_exact_decode(caches, decode_steps=1)
        self.assertEqual(info["merge"], "exact_structural_concat")
        self.assertEqual(tuple(merged.layers[0].keys.shape), (2, 1, 3, 2))

        merged.update(torch.full((2, 1, 1, 2), 7.0), torch.full((2, 1, 1, 2), 17.0), layer_idx=0)
        merged.split_exact_decode_into(caches)

        for cache in caches:
            self.assertEqual(tuple(cache.layers[0].keys.shape), (1, 1, 4, 2))
            self.assertEqual(cache.layers[0].cumulative_length, 4)
        self.assertTrue(torch.equal(caches[0].layers[0].keys[:, :, -1], torch.full((1, 1, 2), 7.0)))
        self.assertTrue(torch.equal(caches[1].layers[0].values[:, :, -1], torch.full((1, 1, 2), 17.0)))

    def test_exact_decode_single_row_merge_returns_cache(self) -> None:
        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        hf_cfg = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=16,
        )
        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=16,
        )
        cache = NativeGemmaRoutedCache(hf_cfg, cfg)

        merged, info = NativeGemmaRoutedCache.merge_exact_decode([cache], decode_steps=1)

        self.assertIs(merged, cache)
        self.assertEqual(info["merge"], "single_row")

    def test_padded_decode_merge_commits_ragged_rows(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        hf_cfg = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=16,
        )
        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=16,
        )
        caches = [NativeGemmaRoutedCache(hf_cfg, cfg) for _ in range(2)]
        lengths = [3, 5]
        for row, (cache, length) in enumerate(zip(caches, lengths)):
            key = torch.full((1, 1, length, 2), float(row + 1))
            value = torch.full((1, 1, length, 2), float(row + 11))
            cache.update(key, value, layer_idx=0)

        merged, info = NativeGemmaRoutedCache.merge_padded_decode(caches, decode_steps=1)
        self.assertEqual(info["merge"], "padded_valid_mask_concat")
        self.assertEqual(tuple(merged.layers[0].keys.shape), (2, 1, 6, 2))
        self.assertEqual(tuple(merged.layers[0].valid_mask.shape), (2, 5))
        layer_info = info["layers"][0]
        self.assertEqual(layer_info["temporary_kv_bytes"], 192)
        self.assertEqual(layer_info["temporary_mask_bytes"], 10)
        self.assertEqual(layer_info["temporary_total_bytes"], 202)
        self.assertEqual(layer_info["copied_kv_bytes"], 128)
        self.assertEqual(layer_info["padded_kv_bytes"], 32)
        self.assertEqual(layer_info["source_padded_kv_bytes"], 32)
        self.assertEqual(layer_info["workspace_extra_padded_kv_bytes"], 0)
        self.assertEqual(layer_info["reserved_decode_kv_bytes"], 32)
        self.assertEqual(layer_info["copied_slots_total"], 8)
        self.assertEqual(layer_info["pad_slots_total"], 2)
        self.assertEqual(layer_info["source_pad_slots_total"], 2)
        self.assertEqual(layer_info["workspace_extra_pad_slots_total"], 0)
        self.assertEqual(layer_info["reserved_decode_slots_total"], 2)
        mask = merged.padded_attention_mask()["sliding_attention"]
        self.assertEqual(tuple(mask.shape), (2, 1, 1, 6))
        self.assertLess(float(mask[0, 0, 0, 3]), 0.0)
        self.assertEqual(float(mask[0, 0, 0, 5]), 0.0)
        shared_key = torch.full((2, 1, 1, 2), 3.0)
        shared_value = torch.full((2, 1, 1, 2), 13.0)
        merged.store_shared_kv(
            layer_idx=0,
            layer_type="sliding_attention",
            key_states=shared_key,
            value_states=shared_value,
        )
        self.assertIs(
            merged.get_shared_kv(layer_idx=0, layer_type="sliding_attention")[0],
            shared_key,
        )

        merged.update(torch.full((2, 1, 1, 2), 7.0), torch.full((2, 1, 1, 2), 17.0), layer_idx=0)
        merged.commit_padded_decode_into(caches)

        self.assertEqual(tuple(caches[0].layers[0].keys.shape), (1, 1, 4, 2))
        self.assertEqual(tuple(caches[1].layers[0].keys.shape), (1, 1, 6, 2))
        self.assertTrue(torch.equal(caches[0].layers[0].keys[:, :, -1], torch.full((1, 1, 2), 7.0)))
        self.assertTrue(torch.equal(caches[1].layers[0].values[:, :, -1], torch.full((1, 1, 2), 17.0)))

    def test_padded_decode_token_pool_covered_sliding_skips_dense_kv(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_runner import DistinctCacheBatchError, NativeGemmaRoutedCache

        hf_cfg = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=16,
        )
        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=16,
        )
        caches = [NativeGemmaRoutedCache(hf_cfg, cfg) for _ in range(2)]
        lengths = [3, 5]
        for row, (cache, length) in enumerate(zip(caches, lengths)):
            key = torch.full((1, 1, length, 2), float(row + 1))
            value = torch.full((1, 1, length, 2), float(row + 11))
            cache.update(key, value, layer_idx=0)

        merged, info = NativeGemmaRoutedCache.merge_padded_decode(
            caches,
            decode_steps=3,
            persistent=True,
            token_pool_covered_layer_types={"sliding_attention"},
        )

        self.assertEqual(info["merge"], "padded_valid_mask_concat")
        layer_info = info["layers"][0]
        self.assertEqual(layer_info["merge"], "token_pool_covered_skip")
        self.assertEqual(layer_info["temporary_total_bytes"], 0)
        self.assertEqual(layer_info["copied_kv_bytes"], 0)
        self.assertEqual(layer_info["source_materialized_slots_max"], 5)
        self.assertIsNone(getattr(merged.layers[0], "keys", None))
        self.assertIsNone(merged.padded_attention_mask()["sliding_attention"])
        self.assertEqual(merged.padded_decode_remaining_capacity(), 3)
        merged.record_token_pool_covered_decode_step()
        self.assertEqual(merged.padded_decode_remaining_capacity(), 2)
        with self.assertRaisesRegex(DistinctCacheBatchError, "token-pool decode"):
            merged.update(
                torch.full((2, 1, 1, 2), 7.0),
                torch.full((2, 1, 1, 2), 17.0),
                layer_idx=0,
            )
        merged.commit_padded_decode_into(caches)
        self.assertEqual(tuple(caches[0].layers[0].keys.shape), (1, 1, 3, 2))
        self.assertEqual(tuple(caches[1].layers[0].keys.shape), (1, 1, 5, 2))

        for cache in caches:
            self.assertEqual(
                cache.release_token_pool_covered_sliding_storage({"sliding_attention"}),
                1,
            )
            self.assertIsNone(cache.layers[0].keys)
            self.assertIsNone(cache.layers[0].values)
        released_merged, released_info = NativeGemmaRoutedCache.merge_padded_decode(
            caches,
            decode_steps=3,
            persistent=True,
            token_pool_covered_layer_types={"sliding_attention"},
        )
        self.assertEqual(released_info["layers"][0]["temporary_total_bytes"], 0)
        self.assertEqual(released_info["layers"][0]["source_materialized_slots_max"], 0)
        self.assertIsNone(released_merged.padded_attention_mask()["sliding_attention"])
        self.assertEqual(released_merged.padded_decode_remaining_capacity(), 3)
        with self.assertRaisesRegex(DistinctCacheBatchError, "released"):
            caches[0].update(
                torch.full((1, 1, 1, 2), 9.0),
                torch.full((1, 1, 1, 2), 19.0),
                layer_idx=0,
            )

    def test_padded_decode_token_pool_covered_full_skips_dense_kv(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_runner import DistinctCacheBatchError, NativeGemmaRoutedCache

        hf_cfg = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("full_attention",),
            sliding_window=8,
        )
        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("full_attention",),
            sink_tokens=1,
            ring_tokens=2,
            pending_tokens=8,
            routed_slots=2,
            reps_per_slot=1,
            span_budget_tokens=2,
            max_span_tokens=2,
        )
        caches = [NativeGemmaRoutedCache(hf_cfg, cfg) for _ in range(2)]
        lengths = [3, 4]
        for row, (cache, length) in enumerate(zip(caches, lengths)):
            key = torch.full((1, 1, length, 2), float(row + 1))
            value = torch.full((1, 1, length, 2), float(row + 11))
            cache.update(key, value, layer_idx=0)

        merged, info = NativeGemmaRoutedCache.merge_padded_decode(
            caches,
            decode_steps=3,
            persistent=True,
            token_pool_covered_layer_types={"full_attention"},
        )

        layer_info = info["layers"][0]
        self.assertEqual(layer_info["merge"], "token_pool_covered_skip")
        self.assertEqual(layer_info["temporary_total_bytes"], 0)
        self.assertEqual(layer_info["copied_kv_bytes"], 0)
        self.assertEqual(layer_info["source_materialized_slots_max"], 4)
        self.assertEqual(layer_info["token_pool_covered_layer_type"], "full_attention")
        self.assertIsNone(getattr(merged.layers[0], "keys", None))
        self.assertIsNone(merged.padded_attention_mask()["full_attention"])
        self.assertEqual(merged.padded_decode_remaining_capacity(), 3)
        merged.record_token_pool_covered_decode_step()
        self.assertEqual(merged.padded_decode_remaining_capacity(), 2)
        with self.assertRaisesRegex(DistinctCacheBatchError, "token-pool decode"):
            merged.update(
                torch.full((2, 1, 1, 2), 7.0),
                torch.full((2, 1, 1, 2), 17.0),
                layer_idx=0,
            )
        merged.commit_padded_decode_into(caches)
        self.assertEqual(tuple(caches[0].layers[0].keys.shape), (1, 1, 3, 2))
        self.assertEqual(tuple(caches[1].layers[0].keys.shape), (1, 1, 4, 2))

    def test_routed_layer_backfills_materialized_readout_to_token_pool(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache
        from wkvm.runner.gemma_token_pool import (
            TokenKVLayerSpec,
            TokenKVPool,
            build_decode_metadata_from_token_slot_rows,
        )

        hf_cfg = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("full_attention",),
            sliding_window=8,
        )
        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("full_attention",),
            sink_tokens=1,
            ring_tokens=1,
            pending_tokens=2,
            routed_slots=2,
            reps_per_slot=1,
            span_budget_tokens=4,
            max_span_tokens=3,
        )
        cache = NativeGemmaRoutedCache(hf_cfg, cfg)
        key = torch.arange(10, dtype=torch.float32).reshape(1, 1, 5, 2)
        value = key + 100
        cache.update(key, value, layer_idx=0)
        layer = cache.layers[0]
        width = layer.materialized_tokens()
        self.assertGreater(width, 0)
        self.assertLessEqual(width, 5)

        pool = TokenKVPool(
            capacity=8,
            layer_specs=[
                TokenKVLayerSpec(layer_id=0, num_kv_heads=1, head_dim=2, dtype=torch.float32),
            ],
            dtype=torch.float32,
        )
        allocated = pool.alloc_slots(8)
        slot_row = allocated[
            torch.tensor([5, 2, 4, 1, 0, 7], dtype=torch.long)
        ][:width]
        self.assertEqual(int(slot_row.numel()), width)

        layer.write_materialized_readout_to_token_pool(pool, slot_row)
        gathered_k, gathered_v = pool.gather_kv(0, slot_row)
        expected_k = layer.keys[0].permute(1, 0, 2).contiguous()
        expected_v = layer.values[0].permute(1, 0, 2).contiguous()
        self.assertTrue(torch.equal(gathered_k, expected_k))
        self.assertTrue(torch.equal(gathered_v, expected_v))

        self.assertFalse(
            layer.release_dense_materialized_storage(
                reserve_steps=layer.route_chunk,
            )
        )
        self.assertIsNotNone(layer.keys)
        self.assertTrue(layer.release_dense_materialized_storage())
        self.assertIsNone(layer.keys)
        self.assertIsNone(layer.values)
        self.assertEqual(layer.materialized_tokens(), width)

        decode_key = torch.tensor([[[[21.0, 22.0]]]])
        decode_value = decode_key + 100
        self.assertTrue(layer.commit_decode_token(decode_key, decode_value))
        self.assertIsNone(layer.keys)
        self.assertIsNone(layer.values)
        self.assertEqual(layer.materialized_tokens(), width + 1)

        extended_slot_row = torch.cat((slot_row, allocated[6:7]))
        layer.write_materialized_readout_to_token_pool(pool, extended_slot_row)
        gathered_k, gathered_v = pool.gather_kv(0, extended_slot_row)
        self.assertTrue(
            torch.equal(gathered_k, layer.keys[0].permute(1, 0, 2).contiguous())
        )
        self.assertTrue(
            torch.equal(gathered_v, layer.values[0].permute(1, 0, 2).contiguous())
        )

        metadata = build_decode_metadata_from_token_slot_rows(
            [slot_row],
            logical_seq_lens=[layer.cumulative_length],
            out_cache_loc=[int(slot_row[-1].item())],
        )
        self.assertEqual(metadata.kv_indices.tolist(), slot_row.tolist())
        self.assertEqual(metadata.logical_seq_lens.tolist(), [layer.cumulative_length])

    def test_routed_release_guard_covers_exact_512_token_fold_horizon(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        hf_cfg = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("full_attention",),
            sliding_window=1024,
        )
        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("full_attention",),
            sink_tokens=16,
            ring_tokens=1024,
            pending_tokens=512,
            routed_slots=8,
            reps_per_slot=2,
            span_budget_tokens=48,
            max_span_tokens=48,
            sliding_window=1024,
        )
        cache = NativeGemmaRoutedCache(hf_cfg, cfg)
        break_mask = [False] * 2560
        break_mask[1151] = True
        cache.set_span_break_mask(break_mask)
        keys = torch.arange(5120, dtype=torch.float32).reshape(1, 1, 2560, 2)
        cache.update(keys, keys + 10_000, layer_idx=0)
        layer = cache.layers[0]

        self.assertEqual(int(layer._pend_k.shape[2]), 384)
        self.assertFalse(layer.release_dense_materialized_storage(reserve_steps=128))
        self.assertIsNotNone(layer.keys)
        self.assertIsNotNone(layer.values)
        self.assertFalse(layer._dense_storage_released)

        later_keys = torch.arange(256, dtype=torch.float32).reshape(1, 1, 128, 2)
        cache.update(later_keys, later_keys + 20_000, layer_idx=0)
        self.assertEqual(layer.cumulative_length, 2688)
        self.assertLess(int(layer._pend_k.shape[2]), layer.route_chunk)
        self.assertIsNotNone(layer.keys)
        self.assertIsNotNone(layer.values)

    def test_large_prefill_bounds_each_route_fold_to_route_chunk(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        hf_cfg = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("full_attention",),
            sliding_window=1024,
        )
        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("full_attention",),
            sink_tokens=16,
            ring_tokens=1024,
            pending_tokens=512,
            routed_slots=8,
            reps_per_slot=2,
            span_budget_tokens=48,
            max_span_tokens=48,
            sliding_window=1024,
        )
        cache = NativeGemmaRoutedCache(hf_cfg, cfg)
        layer = cache.layers[0]
        original_route_fold = layer._route_fold
        fold_widths = []

        def record_route_fold(cut_k, cut_v):
            fold_widths.append(int(cut_k.shape[2]))
            return original_route_fold(cut_k, cut_v)

        layer._route_fold = record_route_fold
        keys = torch.arange(5120, dtype=torch.float32).reshape(1, 1, 2560, 2)
        cache.update(keys, keys + 10_000, layer_idx=0)

        self.assertGreater(len(fold_widths), 1)
        self.assertLessEqual(max(fold_widths), layer.route_chunk)

    def test_persistent_padded_decode_reuses_cache_and_commits_all_steps(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        hf_cfg = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=16,
        )
        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=16,
        )
        caches = [NativeGemmaRoutedCache(hf_cfg, cfg) for _ in range(2)]
        lengths = [3, 5]
        for row, (cache, length) in enumerate(zip(caches, lengths)):
            key = torch.full((1, 1, length, 2), float(row + 1))
            value = torch.full((1, 1, length, 2), float(row + 11))
            cache.update(key, value, layer_idx=0)

        merged, info = NativeGemmaRoutedCache.merge_padded_decode(
            caches,
            decode_steps=3,
            persistent=True,
        )
        self.assertEqual(info["merge"], "padded_valid_mask_concat")
        self.assertEqual(tuple(merged.layers[0].keys.shape), (2, 1, 8, 2))
        self.assertEqual(merged.padded_decode_remaining_capacity(), 3)

        first_mask = merged.padded_attention_mask()["sliding_attention"]
        self.assertEqual(tuple(first_mask.shape), (2, 1, 1, 6))
        self.assertLess(float(first_mask[0, 0, 0, 3]), 0.0)
        self.assertEqual(float(first_mask[0, 0, 0, 5]), 0.0)

        merged.update(torch.full((2, 1, 1, 2), 7.0), torch.full((2, 1, 1, 2), 17.0), layer_idx=0)
        self.assertEqual(merged.padded_decode_remaining_capacity(), 2)
        second_mask = merged.padded_attention_mask()["sliding_attention"]
        self.assertEqual(tuple(second_mask.shape), (2, 1, 1, 7))
        self.assertEqual(float(second_mask[0, 0, 0, 5]), 0.0)
        self.assertEqual(float(second_mask[0, 0, 0, 6]), 0.0)

        merged.update(torch.full((2, 1, 1, 2), 8.0), torch.full((2, 1, 1, 2), 18.0), layer_idx=0)
        merged.commit_padded_decode_into(caches)

        self.assertEqual(tuple(caches[0].layers[0].keys.shape), (1, 1, 5, 2))
        self.assertEqual(tuple(caches[1].layers[0].keys.shape), (1, 1, 7, 2))
        self.assertTrue(torch.equal(caches[0].layers[0].keys[:, :, -2], torch.full((1, 1, 2), 7.0)))
        self.assertTrue(torch.equal(caches[0].layers[0].keys[:, :, -1], torch.full((1, 1, 2), 8.0)))
        self.assertTrue(torch.equal(caches[1].layers[0].values[:, :, -2], torch.full((1, 1, 2), 17.0)))
        self.assertTrue(torch.equal(caches[1].layers[0].values[:, :, -1], torch.full((1, 1, 2), 18.0)))

    def test_persistent_padded_routed_commit_survives_released_source_readouts(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        hf_config = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("full_attention",),
            sliding_window=8,
        )
        config = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("full_attention",),
            sink_tokens=1,
            ring_tokens=2,
            pending_tokens=8,
            routed_slots=2,
            reps_per_slot=1,
            span_budget_tokens=2,
            max_span_tokens=2,
            sliding_window=8,
        )
        caches = [NativeGemmaRoutedCache(hf_config, config) for _ in range(2)]
        for row, cache in enumerate(caches):
            width = row + 3
            keys = torch.full((1, 1, width, 2), float(row + 1))
            cache.update(keys, keys + 10, layer_idx=0)

        merged, _info = NativeGemmaRoutedCache.merge_padded_decode(
            caches,
            decode_steps=2,
            persistent=True,
        )
        for cache in caches:
            self.assertTrue(
                cache.layers[0].release_dense_materialized_storage(reserve_steps=2)
            )
            self.assertIsNone(cache.layers[0].keys)
        merged.update(
            torch.full((2, 1, 1, 2), 7.0),
            torch.full((2, 1, 1, 2), 17.0),
            layer_idx=0,
        )

        merged.commit_padded_decode_into(caches)

        self.assertTrue(all(cache.layers[0]._dense_storage_released for cache in caches))
        self.assertEqual([cache.layers[0].cumulative_length for cache in caches], [4, 5])
        rematerialized, _info = NativeGemmaRoutedCache.merge_padded_decode(
            caches,
            decode_steps=1,
            persistent=True,
        )
        self.assertTrue(all(cache.layers[0].keys is not None for cache in caches))
        self.assertEqual(int(rematerialized.layers[0].keys.shape[0]), 2)

    def test_persistent_padded_sliding_multi_token_commit_skips_update(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        hf_cfg = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=6,
        )
        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=6,
        )
        caches = [NativeGemmaRoutedCache(hf_cfg, cfg) for _ in range(2)]
        lengths = [3, 5]
        for row, (cache, length) in enumerate(zip(caches, lengths)):
            key = torch.full((1, 1, length, 2), float(row + 1))
            value = torch.full((1, 1, length, 2), float(row + 11))
            cache.update(key, value, layer_idx=0)

        merged, info = NativeGemmaRoutedCache.merge_padded_decode(
            caches,
            decode_steps=3,
            persistent=True,
        )
        self.assertEqual(info["merge"], "padded_valid_mask_concat")
        merged.update(
            torch.full((2, 1, 1, 2), 7.0),
            torch.full((2, 1, 1, 2), 17.0),
            layer_idx=0,
        )
        merged.update(
            torch.full((2, 1, 1, 2), 8.0),
            torch.full((2, 1, 1, 2), 18.0),
            layer_idx=0,
        )

        for cache in caches:
            cache.layers[0].update = lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("sliding persistent commit fell back to update")
            )
        merged.commit_padded_decode_into(caches)

        self.assertEqual(tuple(caches[0].layers[0].keys.shape), (1, 1, 5, 2))
        self.assertEqual(tuple(caches[1].layers[0].keys.shape), (1, 1, 5, 2))
        self.assertTrue(torch.equal(caches[0].layers[0].keys[:, :, -2], torch.full((1, 1, 2), 7.0)))
        self.assertTrue(torch.equal(caches[0].layers[0].keys[:, :, -1], torch.full((1, 1, 2), 8.0)))
        self.assertTrue(torch.equal(caches[1].layers[0].values[:, :, -2], torch.full((1, 1, 2), 17.0)))
        self.assertTrue(torch.equal(caches[1].layers[0].values[:, :, -1], torch.full((1, 1, 2), 18.0)))

    def test_static_persistent_padded_decode_keeps_fixed_mask_shape(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        hf_cfg = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=16,
        )
        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=16,
        )
        caches = [NativeGemmaRoutedCache(hf_cfg, cfg) for _ in range(2)]
        lengths = [3, 5]
        for row, (cache, length) in enumerate(zip(caches, lengths)):
            key = torch.full((1, 1, length, 2), float(row + 1))
            value = torch.full((1, 1, length, 2), float(row + 11))
            cache.update(key, value, layer_idx=0)

        merged, info = NativeGemmaRoutedCache.merge_padded_decode(
            caches,
            decode_steps=3,
            persistent=True,
            graph_static=True,
        )
        self.assertEqual(info["merge"], "padded_valid_mask_concat")
        self.assertTrue(merged.static_padded_decode)
        self.assertEqual(tuple(merged.layers[0].keys.shape), (2, 1, 8, 2))

        first_mask = merged.graph_padded_attention_mask()["sliding_attention"]
        self.assertEqual(tuple(first_mask.shape), (2, 1, 1, 8))
        self.assertLess(float(first_mask[0, 0, 0, 3]), 0.0)
        self.assertLess(float(first_mask[0, 0, 0, 5]), 0.0)
        owner_layer = merged.layers[0]
        owner_valid_mask = owner_layer.valid_mask.clone()

        snapshot = merged.snapshot_static_padded_decode_state()
        merged.update(
            torch.full((2, 1, 1, 2), 6.0),
            torch.full((2, 1, 1, 2), 16.0),
            layer_idx=0,
        )
        self.assertEqual(merged.padded_decode_remaining_capacity(), 2)
        merged.restore_static_padded_decode_state(snapshot)
        self.assertEqual(merged.padded_decode_remaining_capacity(), 3)
        merged.record_static_padded_decode_replay()
        self.assertEqual(merged.padded_decode_remaining_capacity(), 2)
        merged.restore_static_padded_decode_state(snapshot)
        merged.set_static_valid_mask_updates_enabled(False)

        updated_keys, updated_values = merged.update(
            torch.full((2, 1, 1, 2), 7.0),
            torch.full((2, 1, 1, 2), 17.0),
            layer_idx=0,
        )
        self.assertEqual(tuple(updated_keys.shape), (2, 1, 8, 2))
        self.assertEqual(tuple(updated_values.shape), (2, 1, 8, 2))
        second_mask = merged.graph_padded_attention_mask()["sliding_attention"]
        self.assertIs(first_mask, second_mask)
        self.assertEqual(tuple(second_mask.shape), (2, 1, 1, 8))
        self.assertEqual(float(second_mask[0, 0, 0, 5]), 0.0)
        self.assertLess(float(second_mask[0, 0, 0, 6]), 0.0)
        self.assertTrue(torch.equal(owner_layer.valid_mask, owner_valid_mask))

        merged.update(torch.full((2, 1, 1, 2), 8.0), torch.full((2, 1, 1, 2), 18.0), layer_idx=0)
        third_mask = merged.graph_padded_attention_mask()["sliding_attention"]
        self.assertEqual(float(third_mask[0, 0, 0, 6]), 0.0)
        self.assertLess(float(third_mask[0, 0, 0, 7]), 0.0)
        self.assertEqual(merged.padded_decode_remaining_capacity(), 1)
        self.assertTrue(torch.equal(owner_layer.valid_mask, owner_valid_mask))

        merged.commit_padded_decode_into(caches)
        self.assertEqual(tuple(caches[0].layers[0].keys.shape), (1, 1, 5, 2))
        self.assertEqual(tuple(caches[1].layers[0].keys.shape), (1, 1, 7, 2))
        self.assertTrue(torch.equal(caches[0].layers[0].keys[:, :, -2], torch.full((1, 1, 2), 7.0)))
        self.assertTrue(torch.equal(caches[0].layers[0].keys[:, :, -1], torch.full((1, 1, 2), 8.0)))
        self.assertTrue(torch.equal(caches[1].layers[0].values[:, :, -2], torch.full((1, 1, 2), 17.0)))
        self.assertTrue(torch.equal(caches[1].layers[0].values[:, :, -1], torch.full((1, 1, 2), 18.0)))

    def test_static_persistent_padded_decode_uses_one_mask_owner_per_type(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        hf_cfg = SimpleNamespace(
            num_hidden_layers=2,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention", "sliding_attention"),
            sliding_window=16,
        )
        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=2,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention", "sliding_attention"),
            sliding_window=16,
        )
        cache = NativeGemmaRoutedCache(hf_cfg, cfg)
        for layer_idx in range(2):
            cache.update(
                torch.full((1, 1, 3, 2), float(layer_idx + 1)),
                torch.full((1, 1, 3, 2), float(layer_idx + 11)),
                layer_idx=layer_idx,
            )

        merged, info = NativeGemmaRoutedCache.merge_padded_decode(
            [cache],
            decode_steps=2,
            persistent=True,
            graph_static=True,
        )
        self.assertEqual(info["merge"], "padded_valid_mask_concat")
        static_layers = [layer for layer in merged.layers if getattr(layer, "_static_width", False)]
        masks = [layer.static_attention_mask() for layer in static_layers]
        owners = [mask for mask in masks if mask is not None]
        self.assertEqual(len(owners), 1)
        self.assertIs(merged.graph_padded_attention_mask()["sliding_attention"], owners[0])

        merged.update(
            torch.full((1, 1, 1, 2), 7.0),
            torch.full((1, 1, 1, 2), 17.0),
            layer_idx=0,
        )
        graph_mask = merged.graph_padded_attention_mask()["sliding_attention"]
        self.assertEqual(float(graph_mask[0, 0, 0, 3]), 0.0)
        self.assertLess(float(graph_mask[0, 0, 0, 4]), 0.0)

    def test_persistent_padded_routed_decode_allows_reserved_route_margin(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        hf_cfg = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("full_attention",),
            sliding_window=8,
        )
        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("full_attention",),
            num_kv_heads=1,
            head_dim=2,
            sink_tokens=1,
            ring_tokens=8,
            pending_tokens=4,
            routed_slots=2,
            sliding_window=8,
        )
        caches = [NativeGemmaRoutedCache(hf_cfg, cfg)]
        caches[0].update(
            torch.full((1, 1, 3, 2), 1.0),
            torch.full((1, 1, 3, 2), 11.0),
            layer_idx=0,
        )

        merged, info = NativeGemmaRoutedCache.merge_padded_decode(
            caches,
            decode_steps=3,
            persistent=True,
        )
        self.assertEqual(info["merge"], "padded_valid_mask_concat")

        for step in range(3):
            merged.update(
                torch.full((1, 1, 1, 2), float(7 + step)),
                torch.full((1, 1, 1, 2), float(17 + step)),
                layer_idx=0,
            )

        merged.commit_padded_decode_into(caches)
        self.assertEqual(tuple(caches[0].layers[0].keys.shape), (1, 1, 6, 2))
        self.assertTrue(torch.equal(caches[0].layers[0].keys[:, :, -1], torch.full((1, 1, 2), 9.0)))

    def test_static_persistent_padded_routed_replay_allows_reserved_route_margin(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        hf_cfg = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("full_attention",),
            sliding_window=8,
        )
        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("full_attention",),
            num_kv_heads=1,
            head_dim=2,
            sink_tokens=1,
            ring_tokens=8,
            pending_tokens=4,
            routed_slots=2,
            sliding_window=8,
        )
        caches = [NativeGemmaRoutedCache(hf_cfg, cfg)]
        caches[0].update(
            torch.full((1, 1, 3, 2), 1.0),
            torch.full((1, 1, 3, 2), 11.0),
            layer_idx=0,
        )

        merged, info = NativeGemmaRoutedCache.merge_padded_decode(
            caches,
            decode_steps=3,
            persistent=True,
            graph_static=True,
        )
        self.assertEqual(info["merge"], "padded_valid_mask_concat")

        for remaining in (2, 1, 0):
            merged.record_static_padded_decode_replay()
            self.assertEqual(merged.padded_decode_remaining_capacity(), remaining)

    def test_padded_decode_workspace_reuses_bucketed_buffers(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache, PaddedDecodeWorkspace

        hf_cfg = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=16,
        )
        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=16,
        )
        workspace = PaddedDecodeWorkspace(width_bucket=8)

        def make_caches():
            caches = [NativeGemmaRoutedCache(hf_cfg, cfg) for _ in range(2)]
            lengths = [3, 5]
            for row, (cache, length) in enumerate(zip(caches, lengths)):
                key = torch.full((1, 1, length, 2), float(row + 1))
                value = torch.full((1, 1, length, 2), float(row + 11))
                cache.update(key, value, layer_idx=0)
            return caches

        caches = make_caches()
        merged, info = NativeGemmaRoutedCache.merge_padded_decode(
            caches,
            decode_steps=1,
            workspace=workspace,
        )
        first_layer_info = info["layers"][0]
        self.assertEqual(first_layer_info["workspace_allocated"], 1)
        self.assertEqual(first_layer_info["workspace_reused"], 0)
        self.assertEqual(first_layer_info["workspace_bypassed"], 0)
        self.assertEqual(tuple(merged.layers[0].keys.shape), (2, 1, 8, 2))
        self.assertEqual(tuple(merged.layers[0].valid_mask.shape), (2, 7))
        self.assertEqual(first_layer_info["source_materialized_slots_max"], 5)
        self.assertEqual(first_layer_info["temporary_past_slots"], 7)
        self.assertEqual(first_layer_info["source_pad_slots_total"], 2)
        self.assertEqual(first_layer_info["workspace_extra_pad_slots_total"], 4)
        mask = merged.padded_attention_mask()["sliding_attention"]
        self.assertEqual(tuple(mask.shape), (2, 1, 1, 8))
        self.assertLess(float(mask[0, 0, 0, 3]), 0.0)
        self.assertLess(float(mask[0, 0, 0, 6]), 0.0)
        self.assertEqual(float(mask[0, 0, 0, 7]), 0.0)
        merged.update(torch.full((2, 1, 1, 2), 7.0), torch.full((2, 1, 1, 2), 17.0), layer_idx=0)
        merged.commit_padded_decode_into(caches)
        self.assertEqual(tuple(caches[0].layers[0].keys.shape), (1, 1, 4, 2))
        self.assertEqual(tuple(caches[1].layers[0].keys.shape), (1, 1, 6, 2))

        second_caches = make_caches()
        second_merged, second_info = NativeGemmaRoutedCache.merge_padded_decode(
            second_caches,
            decode_steps=1,
            workspace=workspace,
        )
        second_layer_info = second_info["layers"][0]
        self.assertEqual(second_layer_info["workspace_allocated"], 0)
        self.assertEqual(second_layer_info["workspace_reused"], 1)
        self.assertEqual(second_layer_info["workspace_bypassed"], 0)
        self.assertEqual(workspace.allocations, 1)
        self.assertEqual(workspace.reuses, 1)
        self.assertEqual(workspace.bypasses, 0)
        self.assertEqual(tuple(second_merged.layers[0].keys.shape), (2, 1, 8, 2))

    def test_padded_decode_workspace_keeps_reserved_slot_until_update(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache, PaddedDecodeWorkspace

        hf_cfg = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=16,
        )
        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=16,
        )
        workspace = PaddedDecodeWorkspace(width_bucket=8)

        def make_caches():
            caches = [NativeGemmaRoutedCache(hf_cfg, cfg) for _ in range(2)]
            lengths = [3, 5]
            for row, (cache, length) in enumerate(zip(caches, lengths)):
                key = torch.full((1, 1, length, 2), float(row + 1))
                value = torch.full((1, 1, length, 2), float(row + 11))
                cache.update(key, value, layer_idx=0)
            return caches

        first_merged, _first_info = NativeGemmaRoutedCache.merge_padded_decode(
            make_caches(),
            decode_steps=1,
            workspace=workspace,
        )
        first_merged.layers[0].keys[:, :, -1, :].fill_(float("nan"))
        first_merged.layers[0].values[:, :, -1, :].fill_(float("nan"))

        caches = make_caches()
        second_merged, second_info = NativeGemmaRoutedCache.merge_padded_decode(
            caches,
            decode_steps=1,
            workspace=workspace,
        )
        self.assertEqual(second_info["layers"][0]["workspace_reused"], 1)
        self.assertFalse(torch.isnan(second_merged.layers[0].keys[:, :, :-1, :]).any().item())
        self.assertFalse(torch.isnan(second_merged.layers[0].values[:, :, :-1, :]).any().item())
        self.assertTrue(torch.isnan(second_merged.layers[0].keys[:, :, -1, :]).all().item())
        self.assertTrue(torch.isnan(second_merged.layers[0].values[:, :, -1, :]).all().item())

        second_merged.update(
            torch.full((2, 1, 1, 2), 7.0),
            torch.full((2, 1, 1, 2), 17.0),
            layer_idx=0,
        )
        self.assertFalse(torch.isnan(second_merged.layers[0].keys[:, :, -1, :]).any().item())
        self.assertFalse(torch.isnan(second_merged.layers[0].values[:, :, -1, :]).any().item())
        second_merged.commit_padded_decode_into(caches)

        self.assertEqual(tuple(caches[0].layers[0].keys.shape), (1, 1, 4, 2))
        self.assertEqual(tuple(caches[1].layers[0].keys.shape), (1, 1, 6, 2))
        self.assertTrue(torch.equal(caches[0].layers[0].keys[:, :, -1], torch.full((1, 1, 2), 7.0)))
        self.assertTrue(torch.equal(caches[1].layers[0].values[:, :, -1], torch.full((1, 1, 2), 17.0)))

    def test_padded_routed_decode_fast_commit_matches_generic_update(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        hf_cfg = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("full_attention",),
            sliding_window=8,
        )
        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("full_attention",),
            sink_tokens=1,
            ring_tokens=2,
            pending_tokens=4,
            routed_slots=2,
            reps_per_slot=1,
            span_budget_tokens=2,
            max_span_tokens=2,
        )

        def tensor(start: int, length: int):
            return torch.arange(start, start + length * 2, dtype=torch.float32).reshape(1, 1, length, 2)

        def make_caches():
            caches = [NativeGemmaRoutedCache(hf_cfg, cfg) for _ in range(2)]
            for row, (cache, length) in enumerate(zip(caches, (3, 4))):
                cache.update(tensor(10 + row * 100, length), tensor(1000 + row * 100, length), layer_idx=0)
            return caches

        generic_caches = make_caches()
        fast_caches = make_caches()
        decode_keys = torch.cat([tensor(500, 1), tensor(600, 1)], dim=0)
        decode_values = torch.cat([tensor(1500, 1), tensor(1600, 1)], dim=0)

        for row, cache in enumerate(generic_caches):
            cache.update(decode_keys[row : row + 1], decode_values[row : row + 1], layer_idx=0)

        merged, info = NativeGemmaRoutedCache.merge_padded_decode(fast_caches, decode_steps=1)
        self.assertEqual(info["merge"], "padded_valid_mask_concat")
        self.assertEqual(info["layers"][0]["pending_tail"], 1)

        for cache in fast_caches:
            cache.layers[0].update = lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("routed padded commit fell back to update")
            )
        merged.update(decode_keys, decode_values, layer_idx=0)
        merged.commit_padded_decode_into(fast_caches)

        for fast_cache, generic_cache in zip(fast_caches, generic_caches):
            fast_layer = fast_cache.layers[0]
            generic_layer = generic_cache.layers[0]
            self.assertEqual(fast_layer.cumulative_length, generic_layer.cumulative_length)
            self.assertEqual(fast_layer._slot_cnt, generic_layer._slot_cnt)
            for attr in (
                "keys",
                "values",
                "_sink_k",
                "_sink_v",
                "_ring_k",
                "_ring_v",
                "_pend_k",
                "_pend_v",
                "_slot_mk",
                "_slot_mv",
            ):
                self.assertTrue(torch.equal(getattr(fast_layer, attr), getattr(generic_layer, attr)), attr)
            self.assertEqual(fast_layer._ring_k.shape[2], 2)
            self.assertLess(fast_layer._pend_k.shape[2], fast_layer.route_chunk)

    def test_padded_routed_decode_multi_token_fast_commit_matches_generic_update(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        hf_cfg = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("full_attention",),
            sliding_window=8,
        )
        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("full_attention",),
            sink_tokens=1,
            ring_tokens=2,
            pending_tokens=8,
            routed_slots=2,
            reps_per_slot=1,
            span_budget_tokens=2,
            max_span_tokens=2,
        )

        def tensor(start: int, length: int):
            return torch.arange(start, start + length * 2, dtype=torch.float32).reshape(1, 1, length, 2)

        def make_caches():
            caches = [NativeGemmaRoutedCache(hf_cfg, cfg) for _ in range(2)]
            for row, (cache, length) in enumerate(zip(caches, (3, 4))):
                cache.update(tensor(10 + row * 100, length), tensor(1000 + row * 100, length), layer_idx=0)
            return caches

        generic_caches = make_caches()
        fast_caches = make_caches()
        decode_keys = torch.cat([tensor(500, 2), tensor(600, 2)], dim=0)
        decode_values = torch.cat([tensor(1500, 2), tensor(1600, 2)], dim=0)

        for row, cache in enumerate(generic_caches):
            cache.update(decode_keys[row : row + 1], decode_values[row : row + 1], layer_idx=0)

        merged, info = NativeGemmaRoutedCache.merge_padded_decode(fast_caches, decode_steps=2, persistent=True)
        self.assertEqual(info["merge"], "padded_valid_mask_concat")
        self.assertEqual(info["layers"][0]["pending_tail"], 1)

        for cache in fast_caches:
            cache.layers[0].update = lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("routed persistent commit fell back to update")
            )
        merged.update(decode_keys[:, :, :1], decode_values[:, :, :1], layer_idx=0)
        merged.update(decode_keys[:, :, 1:], decode_values[:, :, 1:], layer_idx=0)
        merged.commit_padded_decode_into(fast_caches)

        for fast_cache, generic_cache in zip(fast_caches, generic_caches):
            fast_layer = fast_cache.layers[0]
            generic_layer = generic_cache.layers[0]
            self.assertEqual(fast_layer.cumulative_length, generic_layer.cumulative_length)
            self.assertEqual(fast_layer._slot_cnt, generic_layer._slot_cnt)
            for attr in (
                "keys",
                "values",
                "_sink_k",
                "_sink_v",
                "_ring_k",
                "_ring_v",
                "_pend_k",
                "_pend_v",
                "_slot_mk",
                "_slot_mv",
            ):
                self.assertTrue(torch.equal(getattr(fast_layer, attr), getattr(generic_layer, attr)), attr)
            self.assertEqual(fast_layer._ring_k.shape[2], 2)
            self.assertLess(fast_layer._pend_k.shape[2], fast_layer.route_chunk)

    def test_padded_routed_decode_rejects_route_fold_boundary(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_runner import DistinctCacheBatchError, NativeGemmaRoutedCache

        hf_cfg = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("full_attention",),
            sliding_window=8,
        )
        cfg = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("full_attention",),
            sink_tokens=1,
            ring_tokens=1,
            pending_tokens=2,
            routed_slots=2,
            reps_per_slot=1,
            span_budget_tokens=2,
            max_span_tokens=2,
        )
        caches = [NativeGemmaRoutedCache(hf_cfg, cfg) for _ in range(2)]
        for row, cache in enumerate(caches):
            key = torch.full((1, 1, 3, 2), float(row + 1))
            value = torch.full((1, 1, 3, 2), float(row + 11))
            cache.update(key, value, layer_idx=0)

        with self.assertRaisesRegex(DistinctCacheBatchError, "reaches route_chunk"):
            NativeGemmaRoutedCache.merge_padded_decode(caches, decode_steps=1)

        with self.assertRaisesRegex(DistinctCacheBatchError, "reaches route_chunk"):
            NativeGemmaRoutedCache.merge_padded_decode(
                caches,
                decode_steps=1,
                token_pool_covered_layer_types={"full_attention"},
            )

if __name__ == "__main__":
    unittest.main()
