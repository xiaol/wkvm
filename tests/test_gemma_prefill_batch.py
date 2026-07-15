from __future__ import annotations

from types import SimpleNamespace
import unittest


class _RecordingBank:
    def __init__(self, config) -> None:
        self.config = config
        self.ingested = []

    def ingest_positions(self, slots, positions, break_mask=None) -> None:
        self.ingested.append((dict(slots), list(positions), break_mask))


class _OneLayerPrefillModel:
    wkvm_no_hf_transformer_forward = True

    def __init__(self, config) -> None:
        import torch

        self.config = config
        self.device = torch.device("cpu")
        self._parameter = torch.zeros(1, dtype=torch.float32)
        self.calls = []

    def parameters(self):
        yield self._parameter

    def __call__(
        self,
        *,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        **kwargs,
    ):
        import torch

        keys = input_ids.to(torch.float32).reshape(input_ids.shape[0], 1, input_ids.shape[1], 1)
        values = keys + 100
        returned_keys, returned_values = past_key_values.update(
            keys,
            values,
            layer_idx=0,
        )
        layer_type = self.config.layer_types[0]
        past_key_values.store_shared_kv(
            layer_idx=0,
            layer_type=layer_type,
            key_states=returned_keys,
            value_states=returned_values,
        )
        shared_keys, shared_values = past_key_values.get_shared_kv(
            layer_idx=0,
            layer_type=layer_type,
        )
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "position_ids": position_ids.clone(),
                "attention_mask": {
                    key: value.clone() for key, value in attention_mask.items()
                },
                "keys": returned_keys.clone(),
                "values": returned_values.clone(),
                "shared_keys": shared_keys.clone(),
                "shared_values": shared_values.clone(),
                "update_mode": past_key_values.layers[0]._last_update_mode,
            }
        )
        logits = torch.zeros(input_ids.shape[0], 1, 8)
        for row in range(input_ids.shape[0]):
            logits[row, 0, row + 1] = 1
        return SimpleNamespace(logits=logits)


class TestGemmaPrefillBatchRunner(unittest.TestCase):
    def _assert_routed_layer_equal(self, expected, actual) -> None:
        import torch

        for name in (
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
            "_cent",
            "_gmean",
        ):
            expected_tensor = getattr(expected, name)
            actual_tensor = getattr(actual, name)
            if expected_tensor is None:
                self.assertIsNone(actual_tensor)
            elif hasattr(expected_tensor, "data_ptr"):
                self.assertTrue(torch.equal(expected_tensor, actual_tensor), name)
            else:
                self.assertTrue(
                    torch.equal(
                        torch.as_tensor(expected_tensor),
                        torch.as_tensor(actual_tensor),
                    ),
                    name,
                )
        for name in (
            "cumulative_length",
            "_op_cursor",
            "_evicted",
            "_n_active",
            "_gcnt",
            "_slot_cnt",
            "_slot_span_tokens",
            "_bank_span_tokens",
            "_active_span_slots",
        ):
            self.assertEqual(getattr(expected, name), getattr(actual, name), name)
        self.assertEqual(len(expected._slot_spans), len(actual._slot_spans))
        for expected_slot, actual_slot in zip(
            expected._slot_spans,
            actual._slot_spans,
        ):
            self.assertEqual(len(expected_slot), len(actual_slot))
            for expected_span, actual_span in zip(expected_slot, actual_slot):
                self.assertEqual(expected_span["pos"], actual_span["pos"])
                self.assertTrue(torch.equal(expected_span["k"], actual_span["k"]))
                self.assertTrue(torch.equal(expected_span["v"], actual_span["v"]))

    @staticmethod
    def _real_attention_configs():
        from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
        from wkvm.models.gemma import GemmaRoutedSpanConfig

        hf_config = Gemma4TextConfig(
            vocab_size=64,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            hidden_size_per_layer_input=4,
            vocab_size_per_layer_input=64,
            sliding_window=4,
            layer_types=[
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
            num_kv_shared_layers=2,
            attention_dropout=0.0,
            attention_bias=False,
            global_head_dim=4,
            initializer_range=0.2,
        )
        native_config = GemmaRoutedSpanConfig(
            num_hidden_layers=4,
            num_kv_shared_layers=2,
            layer_types=tuple(hf_config.layer_types),
            num_kv_heads=2,
            head_dim=4,
            sliding_window=4,
            sink_tokens=1,
            ring_tokens=1,
            pending_tokens=2,
            routed_slots=2,
            reps_per_slot=1,
            span_budget_tokens=2,
            max_span_tokens=1,
        )
        return hf_config, native_config

    def _assert_real_attention_serial_batch_parity(self, model, native_config) -> None:
        import torch

        from wkvm.runner.gemma_runner import GemmaRoutedSpanRunner, NativeGemmaRoutedCache

        config = model.config
        runner = GemmaRoutedSpanRunner(model, _RecordingBank(native_config))
        prefixes = [[1, 2, 3, 4, 5], [9, 10, 11]]
        tails = [[6, 7, 8], [12, 13, 14]]
        serial_caches = []
        batched_caches = []
        for prefix in prefixes:
            serial_cache = NativeGemmaRoutedCache(config, native_config)
            batched_cache = NativeGemmaRoutedCache(config, native_config)
            runner.prefill_chunk_step(serial_cache, prefix, {}, start_pos=0)
            runner.prefill_chunk_step(batched_cache, prefix, {}, start_pos=0)
            serial_caches.append(serial_cache)
            batched_caches.append(batched_cache)

        serial_logits = [
            runner.prefill_chunk_step(
                serial_caches[row],
                tails[row],
                {},
                start_pos=len(prefixes[row]),
            ).detach()
            for row in range(2)
        ]
        batched_logits = runner.prefill_batch_step(
            batched_caches,
            tails,
            [{}, {}],
            start_positions=[len(prefix) for prefix in prefixes],
        ).detach()

        for row in range(2):
            self.assertTrue(
                torch.allclose(
                    serial_logits[row][0],
                    batched_logits[row],
                    atol=1e-5,
                    rtol=1e-5,
                )
            )
            for serial_layer, batched_layer in zip(
                serial_caches[row].layers,
                batched_caches[row].layers,
            ):
                self.assertEqual(serial_layer.cumulative_length, batched_layer.cumulative_length)

    def test_hf_eager_ragged_shared_kv_matches_serial_prefill(self) -> None:
        import torch
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM

        torch.manual_seed(7)
        hf_config, native_config = self._real_attention_configs()
        hf_config._attn_implementation = "eager"
        model = Gemma4ForCausalLM(hf_config).eval()

        self._assert_real_attention_serial_batch_parity(model, native_config)

    def test_native_boolean_mask_ragged_shared_kv_matches_serial_prefill(self) -> None:
        import torch
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
        from wkvm.runner.gemma_native_forward import NativeGemma4ForCausalLM

        for backend in ("manual_gqa", "sdpa"):
            with self.subTest(backend=backend):
                torch.manual_seed(11)
                hf_config, native_config = self._real_attention_configs()
                hf_model = Gemma4ForCausalLM(hf_config).eval()
                model = NativeGemma4ForCausalLM(
                    hf_model,
                    native_attention_backend=backend,
                ).eval()

                self._assert_real_attention_serial_batch_parity(model, native_config)

    def test_ragged_sliding_rows_get_independent_masks_and_shared_kv(self) -> None:
        import torch

        from wkvm.models.gemma import gemma4_e4b_routed_span_config
        from wkvm.runner.gemma_runner import (
            GemmaRoutedSpanRunner,
            NativeGemmaRoutedCache,
        )

        hf_config = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=4,
        )
        native_config = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=4,
        )
        caches = [
            NativeGemmaRoutedCache(hf_config, native_config),
            NativeGemmaRoutedCache(hf_config, native_config),
        ]
        caches[0].update(
            torch.tensor([[[[10.0], [11.0]]]]),
            torch.tensor([[[[110.0], [111.0]]]]),
            layer_idx=0,
        )
        caches[1].update(
            torch.tensor([[[[20.0]]]]),
            torch.tensor([[[[120.0]]]]),
            layer_idx=0,
        )
        model = _OneLayerPrefillModel(hf_config)
        bank = _RecordingBank(native_config)
        runner = GemmaRoutedSpanRunner(model, bank)

        logits = runner.prefill_batch_step(
            caches,
            [[12, 13], [21, 22]],
            [{"gemma_routed_span": 0}, {"gemma_routed_span": 1}],
            start_positions=[2, 1],
        )

        self.assertEqual(tuple(logits.shape), (2, 1, 8))
        call = model.calls[0]
        self.assertEqual(tuple(call["keys"].shape), (2, 1, 4, 1))
        mask = call["attention_mask"]["sliding_attention"]
        self.assertEqual(tuple(mask.shape), (2, 1, 2, 4))
        visible = mask.reshape(2, 2, 4)
        self.assertEqual(visible[0].tolist(), [[True, True, True, False], [True] * 4])
        self.assertEqual(
            visible[1].tolist(),
            [[True, True, False, False], [True, True, True, False]],
        )
        self.assertEqual(tuple(caches[0].layers[0].keys.shape), (1, 1, 3, 1))
        self.assertEqual(tuple(caches[1].layers[0].keys.shape), (1, 1, 3, 1))
        row0_shared = caches[0].get_shared_kv(layer_idx=0)
        row1_shared = caches[1].get_shared_kv(layer_idx=0)
        self.assertEqual(tuple(row0_shared[0].shape), (1, 1, 4, 1))
        self.assertEqual(tuple(row1_shared[0].shape), (1, 1, 3, 1))
        self.assertEqual(len(bank.ingested), 2)

    def test_homogeneous_sliding_fast_path_matches_serial_across_chunks(self) -> None:
        import torch

        from wkvm.runner.gemma_runner import (
            NativeSlidingWindowLayer,
            _NativeGemmaPrefillBatchLayer,
        )

        serial_layers = [NativeSlidingWindowLayer(4) for _ in range(2)]
        batched_layers = [NativeSlidingWindowLayer(4) for _ in range(2)]
        next_token = 1
        for query_length in (2, 3, 2):
            row0 = torch.arange(
                next_token,
                next_token + query_length,
                dtype=torch.float32,
            ).reshape(1, 1, query_length, 1)
            row1 = row0 + 50
            key_states = torch.cat([row0, row1], dim=0)
            value_states = key_states + 100
            expected_rows = [
                serial_layers[row].update(
                    key_states[row : row + 1],
                    value_states[row : row + 1],
                )
                for row in range(2)
            ]
            batch_layer = _NativeGemmaPrefillBatchLayer(
                batched_layers,
                [
                    layer.get_mask_sizes(query_length)[0]
                    for layer in batched_layers
                ],
            )

            actual_keys, actual_values = batch_layer.update(key_states, value_states)

            self.assertEqual(batch_layer._last_update_mode, "batched_homogeneous")
            self.assertTrue(
                torch.equal(actual_keys, torch.cat([row[0] for row in expected_rows]))
            )
            self.assertTrue(
                torch.equal(actual_values, torch.cat([row[1] for row in expected_rows]))
            )
            for expected, actual in zip(serial_layers, batched_layers):
                self.assertEqual(expected.cumulative_length, actual.cumulative_length)
                self.assertTrue(torch.equal(expected.keys, actual.keys))
                self.assertTrue(torch.equal(expected.values, actual.values))
            next_token += query_length

    def test_deferred_nonfinal_pool_prefill_keeps_homogeneous_sliding_batching(
        self,
    ) -> None:
        import torch

        from wkvm.models.gemma import gemma4_e4b_routed_span_config
        from wkvm.runner.gemma_runner import (
            GemmaRoutedSpanRunner,
            NativeGemmaRoutedCache,
        )
        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenKVLayerSpec,
            TokenKVPool,
            TokenPoolDecodeBackendState,
        )

        hf_config = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=4,
        )
        native_config = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            num_kv_heads=1,
            head_dim=1,
            sliding_window=4,
        )
        caches = [
            NativeGemmaRoutedCache(hf_config, native_config)
            for _ in range(2)
        ]
        pool = TokenKVPool(
            capacity=16,
            layer_specs=[
                TokenKVLayerSpec(
                    layer_id=0,
                    num_kv_heads=1,
                    head_dim=1,
                    dtype=torch.float32,
                )
            ],
            defer_buffer_allocation=True,
        )
        backend = TokenPoolDecodeBackendState(
            table=ReqToTokenTable(max_requests=2, max_context_len=8),
            allocator=pool,
            kv_pool=pool,
            block_size=4,
        )
        requests = [SimpleNamespace(req_id=f"row-{row}") for row in range(2)]
        for request, cache in zip(requests, caches):
            self.assertIsNone(
                backend.prepare_authoritative_prefill(
                    request,
                    3,
                    expected_length=0,
                    cache=cache,
                    sliding_window=4,
                    final_prefill=False,
                )
            )

        model = _OneLayerPrefillModel(hf_config)
        runner = GemmaRoutedSpanRunner(model, _RecordingBank(native_config))
        runner.prefill_batch_step(
            caches,
            [[1, 2, 3], [4, 5, 6]],
            [{"gemma_routed_span": 0}, {"gemma_routed_span": 1}],
            start_positions=[0, 0],
        )

        self.assertEqual(model.calls[0]["update_mode"], "batched_homogeneous")
        for request, cache in zip(requests, caches):
            result = backend.commit_prefill_tokens(
                request,
                3,
                expected_length=0,
                cache=cache,
                sliding_window=4,
                final_prefill=False,
                defer_kv=True,
            )
            self.assertTrue(result.deferred)
            self.assertTrue(
                backend.table.slots_for(request.req_id).eq(-1).all().item()
            )
        self.assertEqual(pool.allocated_count, 0)
        self.assertEqual(pool.kv_set_calls, 0)

    def test_ragged_sliding_state_uses_serial_fallback(self) -> None:
        import torch

        from wkvm.runner.gemma_runner import (
            NativeSlidingWindowLayer,
            _NativeGemmaPrefillBatchLayer,
        )

        layers = [NativeSlidingWindowLayer(4) for _ in range(2)]
        layers[0].update(
            torch.tensor([[[[1.0], [2.0]]]]),
            torch.tensor([[[[101.0], [102.0]]]]),
        )
        layers[1].update(
            torch.tensor([[[[3.0]]]]),
            torch.tensor([[[[103.0]]]]),
        )
        key_states = torch.tensor([[[[4.0]]], [[[5.0]]]])
        value_states = key_states + 100
        batch_layer = _NativeGemmaPrefillBatchLayer(
            layers,
            [layer.get_mask_sizes(1)[0] for layer in layers],
        )

        keys, values = batch_layer.update(key_states, value_states)

        self.assertEqual(batch_layer._last_update_mode, "serial")
        self.assertEqual(tuple(keys.shape), (2, 1, 3, 1))
        self.assertEqual(tuple(values.shape), (2, 1, 3, 1))
        self.assertEqual(keys[1, 0, :, 0].tolist(), [3.0, 5.0, 0.0])

    def test_homogeneous_routed_fast_path_matches_serial_across_folds(self) -> None:
        import torch

        from wkvm.models.gemma import GemmaRoutedSpanConfig
        from wkvm.runner.gemma_runner import (
            NativeGemmaRoutedCache,
            _NativeGemmaPrefillBatchLayer,
        )

        hf_config = SimpleNamespace(
            num_hidden_layers=2,
            num_kv_shared_layers=0,
            layer_types=("full_attention", "full_attention"),
            sliding_window=8,
        )
        native_config = GemmaRoutedSpanConfig(
            num_hidden_layers=2,
            num_kv_shared_layers=0,
            layer_types=("full_attention", "full_attention"),
            num_kv_heads=1,
            head_dim=2,
            sink_tokens=1,
            ring_tokens=2,
            pending_tokens=3,
            routed_slots=2,
            reps_per_slot=1,
            span_budget_tokens=4,
            max_span_tokens=2,
            sliding_window=8,
        )
        serial_caches = [
            NativeGemmaRoutedCache(hf_config, native_config)
            for _ in range(2)
        ]
        batched_caches = [
            NativeGemmaRoutedCache(hf_config, native_config)
            for _ in range(2)
        ]
        break_mask = [False, True, False, False, True, False] * 4
        for cache in serial_caches + batched_caches:
            cache.set_span_break_mask(break_mask)

        next_token = 1
        for query_length in (5, 4, 3):
            row0 = torch.arange(
                next_token,
                next_token + query_length * 2,
                dtype=torch.float32,
            ).reshape(1, 1, query_length, 2)
            for layer_idx in range(2):
                layer_row0 = row0 + layer_idx * 100
                key_states = torch.cat([layer_row0, layer_row0 + 50], dim=0)
                value_states = torch.cat(
                    [layer_row0 + 10, (layer_row0 + 10) * 3],
                    dim=0,
                )
                expected_rows = [
                    serial_caches[row].layers[layer_idx].update(
                        key_states[row : row + 1],
                        value_states[row : row + 1],
                    )
                    for row in range(2)
                ]
                batched_layers = [
                    cache.layers[layer_idx] for cache in batched_caches
                ]
                batch_layer = _NativeGemmaPrefillBatchLayer(
                    batched_layers,
                    [
                        layer.get_mask_sizes(query_length)[0]
                        for layer in batched_layers
                    ],
                )

                actual_keys, actual_values = batch_layer.update(
                    key_states,
                    value_states,
                )

                self.assertEqual(
                    batch_layer._last_update_mode,
                    "batched_homogeneous",
                )
                self.assertTrue(
                    torch.equal(
                        actual_keys,
                        torch.cat([row[0] for row in expected_rows]),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        actual_values,
                        torch.cat([row[1] for row in expected_rows]),
                    )
                )
                for expected_cache, actual_cache in zip(
                    serial_caches,
                    batched_caches,
                ):
                    self._assert_routed_layer_equal(
                        expected_cache.layers[layer_idx],
                        actual_cache.layers[layer_idx],
                    )
            for cache in serial_caches + batched_caches:
                self.assertEqual(cache.layers[0].coord.pending_operations, 0)
            next_token += query_length * 2

    def test_routed_rows_fold_with_independent_value_features(self) -> None:
        import torch

        from wkvm.models.gemma import gemma4_e4b_routed_span_config
        from wkvm.runner.gemma_runner import GemmaRoutedSpanRunner, NativeGemmaRoutedCache

        hf_config = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("full_attention",),
            sliding_window=8,
        )
        native_config = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("full_attention",),
            sink_tokens=1,
            ring_tokens=1,
            pending_tokens=2,
            routed_slots=2,
            reps_per_slot=1,
            span_budget_tokens=4,
            max_span_tokens=2,
        )
        caches = [
            NativeGemmaRoutedCache(hf_config, native_config),
            NativeGemmaRoutedCache(hf_config, native_config),
        ]
        model = _OneLayerPrefillModel(hf_config)
        runner = GemmaRoutedSpanRunner(model, _RecordingBank(native_config))

        runner.prefill_batch_step(
            caches,
            [[1, 2, 3, 4, 5], [50, 40, 30, 20, 10]],
            [{"gemma_routed_span": 0}, {"gemma_routed_span": 1}],
            start_positions=[0, 0],
        )

        first = caches[0].layers[0]
        second = caches[1].layers[0]
        self.assertEqual(int(first._pend_k.shape[2]), int(second._pend_k.shape[2]))
        self.assertEqual(first.cumulative_length, 5)
        self.assertEqual(second.cumulative_length, 5)
        self.assertFalse(torch.equal(first._slot_mv, second._slot_mv))
        self.assertEqual(int(first.keys.shape[0]), 1)
        self.assertEqual(int(second.keys.shape[0]), 1)


class _EngineCache:
    def __init__(self) -> None:
        self.mutations = 0


class _EngineBatchRunner:
    def __init__(self, *, fail_batch: bool = False) -> None:
        self.fail_batch = fail_batch
        self.batch_calls = 0
        self.serial_calls = 0
        self.last_caches = []

    def build_cache(self, slots):
        return _EngineCache()

    def prefill_chunk_step(self, *args, **kwargs):
        self.serial_calls += 1
        raise AssertionError("batched work must not retry serially")

    def prefill_batch_step(
        self,
        caches,
        token_id_rows,
        slots_by_row,
        *,
        start_positions,
        break_masks,
    ):
        import torch

        self.batch_calls += 1
        self.last_caches = list(caches)
        for cache in caches:
            cache.mutations += 1
        if self.fail_batch:
            raise RuntimeError("synthetic mutating batch failure")
        logits = torch.zeros(len(caches), 1, 16)
        for row in range(len(caches)):
            logits[row, 0, row + 3] = 1
        return logits


class _EngineModel:
    device = "cpu"


class TestGemmaPrefillBatchEngine(unittest.TestCase):
    @staticmethod
    def _engine(*, fail_batch: bool = False, token_pool: bool = False):
        from wkvm.core.scheduler import SchedulerConfig
        from wkvm.gemma_engine import GemmaNativeEngine
        from wkvm.models.gemma import gemma4_e4b_routed_span_config

        config = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
        )
        engine = GemmaNativeEngine(
            model=_EngineModel(),
            config=config,
            num_slots=2,
            prefill_microbatch_rows=2,
            scheduler_config=SchedulerConfig(
                max_tokens_per_step=8,
                max_running_requests=2,
                max_tokens_per_request_per_step=4,
            ),
            enable_token_pool_metadata=token_pool,
            token_pool_capacity=16 if token_pool else None,
        )
        runner = _EngineBatchRunner(fail_batch=fail_batch)
        engine.runner = runner
        return engine, runner

    def test_two_prompts_use_one_prefill_model_call(self) -> None:
        from wkvm.core.request import Request, RequestStatus

        engine, runner = self._engine()
        requests = [
            Request(prompt_token_ids=[1, 2, 3, 4], max_new_tokens=1, req_id="a"),
            Request(prompt_token_ids=[5, 6, 7, 8], max_new_tokens=1, req_id="b"),
        ]
        for request in requests:
            engine.add_request(request)

        finished = engine.step()

        self.assertEqual(runner.batch_calls, 1)
        self.assertEqual(runner.serial_calls, 0)
        self.assertEqual(engine.metrics.prefill_calls, 2)
        self.assertEqual(engine.metrics.prefill_model_calls, 1)
        self.assertEqual(engine.metrics.batched_prefill_model_calls, 1)
        self.assertEqual(engine.metrics.batched_prefill_rows, 2)
        self.assertEqual(engine.metrics.max_prefill_batch_rows, 2)
        self.assertEqual([request.output_token_ids for request in requests], [[3], [4]])
        self.assertEqual(len(finished), 2)
        self.assertTrue(all(request.status is RequestStatus.FINISHED_LENGTH for request in requests))

    def test_mutating_batch_failure_never_retries_serially(self) -> None:
        from wkvm.core.request import Request, RequestStatus

        engine, runner = self._engine(fail_batch=True)
        requests = [
            Request(prompt_token_ids=[1, 2, 3, 4], max_new_tokens=1, req_id="a"),
            Request(prompt_token_ids=[5, 6, 7, 8], max_new_tokens=1, req_id="b"),
        ]
        for request in requests:
            engine.add_request(request)
        released = []
        engine._token_pool_release_request = released.append

        with self.assertRaisesRegex(RuntimeError, "synthetic mutating batch failure"):
            engine.step()

        self.assertEqual(runner.batch_calls, 1)
        self.assertEqual(runner.serial_calls, 0)
        self.assertEqual([cache.mutations for cache in runner.last_caches], [1, 1])
        self.assertTrue(all(request.status is RequestStatus.FINISHED_ERROR for request in requests))
        self.assertTrue(all(request.slots == {} for request in requests))
        self.assertEqual(engine._caches, {})
        self.assertEqual(engine.arena.num_free_slots(), 2)
        self.assertEqual(released, ["a", "b"])

    def test_partial_token_pool_commit_failure_releases_every_request(self) -> None:
        from wkvm.core.request import Request, RequestStatus

        engine, runner = self._engine(token_pool=True)
        requests = [
            Request(prompt_token_ids=[1, 2, 3, 4], max_new_tokens=1, req_id="a"),
            Request(prompt_token_ids=[5, 6, 7, 8], max_new_tokens=1, req_id="b"),
        ]
        for request in requests:
            engine.add_request(request)

        backend = engine._token_pool_decode_backend
        allocator = engine._token_slot_allocator
        self.assertIsNotNone(backend)
        self.assertIsNotNone(allocator)
        original_commit = backend.commit_prefill_tokens
        committed = []

        def fail_after_second_commit(request, *args, **kwargs):
            result = original_commit(request, *args, **kwargs)
            committed.append(str(request.req_id))
            if len(committed) == 2:
                raise RuntimeError("synthetic post-token-pool-commit failure")
            return result

        backend.commit_prefill_tokens = fail_after_second_commit

        with self.assertRaisesRegex(RuntimeError, "post-token-pool-commit failure"):
            engine.step()

        self.assertEqual(committed, ["a", "b"])
        self.assertEqual(runner.batch_calls, 1)
        self.assertEqual(runner.serial_calls, 0)
        self.assertTrue(all(request.status is RequestStatus.FINISHED_ERROR for request in requests))
        self.assertTrue(all(request.slots == {} for request in requests))
        self.assertFalse(backend.has_request("a"))
        self.assertFalse(backend.has_request("b"))
        self.assertEqual(allocator.allocated_count, 0)
        self.assertEqual(engine._caches, {})
        self.assertEqual(engine.arena.num_free_slots(), 2)


if __name__ == "__main__":
    unittest.main()
