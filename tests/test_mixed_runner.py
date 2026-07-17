import unittest
from types import SimpleNamespace

import torch

from wkvm.runner.gemma_native_forward import NativeGemma4ForCausalLM
from wkvm.runner.gemma_runner import (
    GemmaRoutedSpanRunner,
    NativeSlidingWindowLayer,
    NativeGemmaRoutedCache,
    _NativeGemmaPrefillBatchCache,
)
from wkvm.core.mixed_batch import MixedBatchMetadata, MixedBatchRow
from wkvm.core.request import Request
from wkvm.core.scheduler import SchedulerOutput
from wkvm.gemma_engine import GemmaNativeEngine
from wkvm.models.gemma import gemma4_e4b_routed_span_config


class _FakeCache:
    def __init__(self, hf_config, layer):
        self.hf_config = hf_config
        self.native_config = SimpleNamespace()
        self.layers = [layer]
        self.shared = {}

    def store_shared_kv(self, *, layer_idx, layer_type, key_states, value_states):
        self.shared[(int(layer_idx), layer_type)] = (key_states, value_states)

    def clear_shared_kv_store(self):
        self.shared.clear()


class TestMixedPrefillBatchCache(unittest.TestCase):
    def test_ragged_query_lengths_slice_writes_and_mask_padding(self):
        config = SimpleNamespace(
            layer_types=("sliding_attention",),
            sliding_window=8,
        )
        decode_layer = NativeSlidingWindowLayer(8, layer_id=0)
        prefill_layer = NativeSlidingWindowLayer(8, layer_id=0)
        # Seed different histories: the first row is a decode continuation and
        # the second row is a new three-token prefill.
        decode_layer.update(
            torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2),
            torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2),
        )
        decode_cache = _FakeCache(config, decode_layer)
        prefill_cache = _FakeCache(config, prefill_layer)
        batch_cache = _NativeGemmaPrefillBatchCache(
            [decode_cache, prefill_cache],
            query_lengths=[1, 3],
            start_positions=[2, 0],
            device="cpu",
            boolean_attention_mask=True,
            mask_dtype=torch.float32,
        )

        self.assertEqual(batch_cache.max_query_length, 3)
        mask = batch_cache.attention_mask["sliding_attention"]
        self.assertEqual(tuple(mask.shape), (2, 1, 3, 3))
        # Row zero has one valid query token; row one has three.  Padded
        # queries remain numerically defined through key zero only.
        self.assertEqual(mask[0, 0, 0].tolist(), [True, True, True])
        self.assertEqual(mask[0, 0, 1].tolist(), [True, False, False])
        self.assertEqual(mask[1, 0, 0].tolist(), [True, False, False])
        self.assertEqual(mask[1, 0, 2].tolist(), [True, True, True])

        key_states = torch.tensor(
            [
                [[[10.0, 10.0], [99.0, 99.0], [99.0, 99.0]]],
                [[[20.0, 20.0], [21.0, 21.0], [22.0, 22.0]]],
            ]
        )
        value_states = key_states + 100.0
        merged_keys, merged_values = batch_cache.update(
            key_states,
            value_states,
            layer_idx=0,
        )

        self.assertEqual(tuple(merged_keys.shape), (2, 1, 3, 2))
        self.assertEqual(tuple(merged_values.shape), (2, 1, 3, 2))
        self.assertEqual(decode_layer.get_seq_length(), 3)
        self.assertEqual(prefill_layer.get_seq_length(), 3)
        self.assertEqual(decode_layer.keys[0, 0, -1].tolist(), [10.0, 10.0])
        self.assertEqual(
            prefill_layer.keys[0, 0].tolist(),
            [[20.0, 20.0], [21.0, 21.0], [22.0, 22.0]],
        )
        self.assertEqual(prefill_layer.keys[0, 0, -1].tolist(), [22.0, 22.0])

    def test_token_pool_decode_row_skips_released_dense_cache_write(self):
        config = SimpleNamespace(
            layer_types=("sliding_attention",),
            sliding_window=8,
        )
        decode_layer = NativeSlidingWindowLayer(8, layer_id=0)
        prefill_layer = NativeSlidingWindowLayer(8, layer_id=0)
        decode_layer.update(
            torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2),
            torch.arange(4, dtype=torch.float32).reshape(1, 1, 2, 2),
        )
        decode_layer.keys = None
        decode_layer.values = None
        decode_layer._dense_storage_released = True
        batch_cache = _NativeGemmaPrefillBatchCache(
            [
                _FakeCache(config, decode_layer),
                _FakeCache(config, prefill_layer),
            ],
            query_lengths=[1, 3],
            start_positions=[2, 0],
            device="cpu",
            boolean_attention_mask=True,
            mask_dtype=torch.float32,
            token_pool_decode_rows={0},
        )
        key_states = torch.tensor(
            [
                [[[10.0, 10.0], [99.0, 99.0], [99.0, 99.0]]],
                [[[20.0, 20.0], [21.0, 21.0], [22.0, 22.0]]],
            ]
        )

        merged_keys, _ = batch_cache.update(
            key_states,
            key_states + 100.0,
            layer_idx=0,
        )

        self.assertEqual(decode_layer.get_seq_length(), 2)
        self.assertIsNone(decode_layer.keys)
        self.assertIsNone(decode_layer.values)
        self.assertEqual(prefill_layer.get_seq_length(), 3)
        self.assertEqual(merged_keys[0, 0, 0].tolist(), [10.0, 10.0])
        self.assertTrue(torch.equal(merged_keys[0, :, 1:], torch.zeros_like(merged_keys[0, :, 1:])))


class TestMixedLogitGather(unittest.TestCase):
    def test_native_model_gathers_one_logit_row_per_query_length(self):
        model = object.__new__(NativeGemma4ForCausalLM)
        model.config = SimpleNamespace(
            num_hidden_layers=1,
            final_logit_softcapping=None,
        )
        model.num_layers = 1
        hidden = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
        model.text_prefix = lambda **kwargs: SimpleNamespace(
            hidden_states=hidden,
            past_key_values=kwargs.get("past_key_values"),
            shared_kv_states={},
        )
        model.lm_head = lambda values: values

        output = model.forward(
            input_ids=torch.ones((2, 3), dtype=torch.long),
            position_ids=torch.zeros((2, 3), dtype=torch.long),
            past_key_values=object(),
            use_cache=True,
            logits_to_keep=1,
            wkvm_logits_indices=torch.tensor([0, 2]),
        )

        self.assertEqual(tuple(output.logits.shape), (2, 1, 4))
        self.assertTrue(torch.equal(output.logits[0, 0], hidden[0, 0]))
        self.assertTrue(torch.equal(output.logits[1, 0], hidden[1, 2]))


class _MixedRunnerModel:
    wkvm_no_hf_transformer_forward = True

    def __init__(self, config):
        self.config = config
        self.device = torch.device("cpu")
        self.dtype = torch.float32
        self.parameter = torch.zeros(1, dtype=self.dtype)
        self.calls = []

    def parameters(self):
        yield self.parameter

    def __call__(self, *, input_ids, position_ids, attention_mask, past_key_values, **kwargs):
        keys = input_ids.to(torch.float32).reshape(
            input_ids.shape[0], 1, input_ids.shape[1], 1
        )
        values = keys + 100
        returned_keys, returned_values = past_key_values.update(
            keys,
            values,
            layer_idx=0,
        )
        past_key_values.store_shared_kv(
            layer_idx=0,
            layer_type="sliding_attention",
            key_states=returned_keys,
            value_states=returned_values,
        )
        self.calls.append(
            {
                "input_ids": input_ids.clone(),
                "position_ids": position_ids.clone(),
                "attention_mask": attention_mask["sliding_attention"].clone(),
                "kwargs": dict(kwargs),
            }
        )
        # Return one deterministic logit row per request.  The runner's
        # sampling layer only needs the row count and final-token gather.
        logits = torch.zeros(input_ids.shape[0], 1, 8)
        for row in range(input_ids.shape[0]):
            logits[row, 0, row + 1] = 1
        return SimpleNamespace(logits=logits)


class _MixedRunnerBank:
    def __init__(self, config):
        self.config = config
        self.ingested = []

    def ingest_positions(self, slots, positions, break_mask=None):
        self.ingested.append((dict(slots), list(positions), break_mask))


class TestMixedRunnerStep(unittest.TestCase):
    def test_native_runner_executes_decode_and_prefill_rows_together(self):
        hf_config = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=8,
        )
        native_config = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            num_kv_heads=1,
            head_dim=1,
            sliding_window=8,
        )
        model = _MixedRunnerModel(hf_config)
        bank = _MixedRunnerBank(native_config)
        runner = GemmaRoutedSpanRunner(model, bank)
        caches = [
            NativeGemmaRoutedCache(hf_config, native_config),
            NativeGemmaRoutedCache(hf_config, native_config),
        ]
        caches[0].layers[0].update(
            torch.tensor([[[[1.0], [2.0]]]]),
            torch.tensor([[[[101.0], [102.0]]]]),
        )
        metadata = MixedBatchMetadata.from_rows(
            [
                MixedBatchRow("decode", 2, 1, 2, 3),
                MixedBatchRow("prefill", 0, 3, 3, 3, initial=True),
            ]
        )
        logits = runner.mixed_batch_step(
            caches,
            [7, 8, 9, 10],
            [{}, {}],
            metadata=metadata,
            start_positions=[2, 0],
        )

        self.assertEqual(tuple(logits.shape), (2, 1, 8))
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(tuple(model.calls[0]["input_ids"].shape), (2, 3))
        self.assertEqual(model.calls[0]["input_ids"].tolist(), [[7, 7, 7], [8, 9, 10]])
        self.assertEqual([cache.layers[0].get_seq_length() for cache in caches], [3, 3])
        self.assertEqual(
            bank.ingested,
            [({}, [2], None), ({}, [0, 1, 2], None)],
        )

    def test_native_runner_passes_mixed_token_pool_context_and_skips_decode_cache(self):
        hf_config = SimpleNamespace(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=8,
        )
        native_config = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            num_kv_heads=1,
            head_dim=1,
            sliding_window=8,
        )
        model = _MixedRunnerModel(hf_config)
        runner = GemmaRoutedSpanRunner(model, _MixedRunnerBank(native_config))
        caches = [
            NativeGemmaRoutedCache(hf_config, native_config),
            NativeGemmaRoutedCache(hf_config, native_config),
        ]
        decode_layer = caches[0].layers[0]
        decode_layer.update(
            torch.tensor([[[[1.0], [2.0]]]]),
            torch.tensor([[[[101.0], [102.0]]]]),
        )
        decode_layer.keys = None
        decode_layer.values = None
        decode_layer._dense_storage_released = True
        metadata = MixedBatchMetadata.from_rows(
            [
                MixedBatchRow("decode", 2, 1, 2, 3),
                MixedBatchRow("prefill", 0, 3, 3, 3, initial=True),
            ]
        )
        context = SimpleNamespace(
            q_lens=(1, 3),
            decode_row_indices=(0,),
        )

        runner.mixed_batch_step(
            caches,
            [7, 8, 9, 10],
            [{}, {}],
            metadata=metadata,
            start_positions=[2, 0],
            token_pool_decode=context,
        )

        self.assertIs(
            model.calls[0]["kwargs"]["wkvm_token_pool_decode"],
            context,
        )
        self.assertEqual(caches[0].layers[0].get_seq_length(), 2)
        self.assertEqual(caches[1].layers[0].get_seq_length(), 3)
        self.assertEqual(runner.last_mixed_batch_info["merge"], "ragged_mixed_token_pool")
        self.assertEqual(runner.last_mixed_batch_info["token_pool_decode_rows"], 1)


class TestMixedEngineOptIn(unittest.TestCase):
    def test_execute_uses_one_call_for_q1_decode_and_qn_prefill(self):
        config = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=8,
        )

        class Model:
            wkvm_no_hf_transformer_forward = True
            device = torch.device("cpu")
            dtype = torch.float32

            def parameters(self):
                yield torch.zeros(1, dtype=torch.float32)

        class Cache:
            def state_bytes(self):
                return 0

        class Runner:
            model = Model()

            def __init__(self):
                self.calls = []

            def build_cache(self, slots):
                return Cache()

            def mixed_batch_step(
                self,
                caches,
                token_ids,
                slots_by_row,
                *,
                metadata,
                start_positions,
                break_masks,
            ):
                self.calls.append(
                    {
                        "q_lens": metadata.q_lens,
                        "token_ids": tuple(token_ids),
                        "start_positions": tuple(start_positions),
                    }
                )
                logits = torch.zeros(len(caches), 1, 8)
                logits[:, 0, 3] = 1
                return logits

        engine = GemmaNativeEngine(
            model=Model(),
            config=config,
            num_slots=2,
            enable_mixed_batch=True,
            enable_token_pool_metadata=False,
        )
        runner = Runner()
        engine.runner = runner  # type: ignore[assignment]
        prefill = Request([1, 2, 3], max_new_tokens=1, req_id="prefill")
        decode = Request([4], max_new_tokens=2, req_id="decode")
        decode.num_computed_tokens = 1
        decode.output_token_ids = [9]
        decode.slots = {"gemma_routed_span": 1}
        engine._caches[decode.req_id] = Cache()  # type: ignore[assignment]
        engine.scheduler.requests = {
            prefill.req_id: prefill,
            decode.req_id: decode,
        }

        sampled = engine._execute(
            SchedulerOutput(
                num_scheduled_tokens={
                    prefill.req_id: 3,
                    decode.req_id: 1,
                }
            )
        )

        self.assertEqual(sampled, {"prefill": [3], "decode": [3]})
        self.assertEqual(len(runner.calls), 1)
        self.assertEqual(runner.calls[0]["q_lens"], (3, 1))
        self.assertEqual(runner.calls[0]["token_ids"], (1, 2, 3, 9))
        self.assertEqual(engine.metrics.mixed_batch_model_calls, 1)
        self.assertEqual(engine.metrics.mixed_batch_rows, 2)
        self.assertEqual(engine.metrics.mixed_batch_opportunities, 1)
        self.assertEqual(engine.metrics.mixed_batch_fallbacks, 0)
        self.assertEqual(engine.metrics.decode_rows, 1)
        self.assertEqual(engine.metrics.max_decode_batch_rows, 1)
        self.assertEqual(engine.metrics.max_decode_model_batch_rows, 1)
        self.assertEqual(engine.metrics.batched_decode_model_calls, 0)

    def test_execute_keeps_token_pool_metadata_in_the_mixed_call(self):
        config = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=8,
        )

        class Model:
            wkvm_no_hf_transformer_forward = True
            device = torch.device("cpu")
            dtype = torch.float32

            def parameters(self):
                yield torch.zeros(1, dtype=torch.float32)

        class Cache:
            def state_bytes(self):
                return 0

        class Runner:
            model = Model()

            def __init__(self):
                self.calls = []

            def build_cache(self, slots):
                return Cache()

            def mixed_batch_step(
                self,
                caches,
                token_ids,
                slots_by_row,
                *,
                metadata,
                start_positions,
                break_masks,
                token_pool_decode=None,
            ):
                self.calls.append(
                    {
                        "q_lens": metadata.q_lens,
                        "decode_rows": token_pool_decode.decode_row_indices,
                        "context_q_lens": token_pool_decode.q_lens,
                    }
                )
                logits = torch.zeros(len(caches), 1, 8)
                logits[:, 0, 4] = 1
                return logits

        class DecodeContext:
            def covered_decode_layer_types(self):
                return frozenset({"sliding_attention"})

        engine = GemmaNativeEngine(
            model=Model(),
            config=config,
            num_slots=2,
            enable_mixed_batch=True,
            enable_token_pool_metadata=False,
        )
        runner = Runner()
        engine.runner = runner  # type: ignore[assignment]
        engine.enable_token_pool_metadata = True
        engine.enable_token_pool_attention = True
        engine._token_kv_pool = SimpleNamespace(layer_specs={0: object()})
        engine._token_pool_decode_backend = object()  # type: ignore[assignment]
        prepared_decode = (object(),)
        committed = []
        engine._token_pool_prepare_authoritative_prefill = (  # type: ignore[method-assign]
            lambda *_args, **_kwargs: None
        )
        engine._token_pool_prepare_decode_model_batch = (  # type: ignore[method-assign]
            lambda _reqs: prepared_decode
        )
        engine._token_pool_decode_context = (  # type: ignore[method-assign]
            lambda _prepared: DecodeContext()
        )
        engine._token_pool_layer_plan = (  # type: ignore[method-assign]
            lambda: SimpleNamespace(
                layer_type_by_layer_id={0: "sliding_attention"}
            )
        )
        engine._token_pool_commit_prefill_tokens = (  # type: ignore[method-assign]
            lambda req, _n, **_kwargs: committed.append(("prefill", req.req_id))
        )
        engine._token_pool_commit_decode_reservations = (  # type: ignore[method-assign]
            lambda prepared: committed.append(("decode", prepared))
        )
        engine._token_pool_release_prefill_sliding_storage = (  # type: ignore[method-assign]
            lambda _cache: None
        )
        prefill = Request([1, 2, 3], max_new_tokens=1, req_id="prefill")
        decode = Request([4], max_new_tokens=2, req_id="decode")
        decode.num_computed_tokens = 1
        decode.output_token_ids = [9]
        decode.slots = {"gemma_routed_span": 1}
        engine._caches[decode.req_id] = Cache()  # type: ignore[assignment]
        engine.scheduler.requests = {
            prefill.req_id: prefill,
            decode.req_id: decode,
        }

        sampled = engine._execute(
            SchedulerOutput(
                num_scheduled_tokens={
                    prefill.req_id: 3,
                    decode.req_id: 1,
                }
            )
        )

        self.assertEqual(sampled, {"prefill": [4], "decode": [4]})
        self.assertEqual(runner.calls[0]["q_lens"], (3, 1))
        self.assertEqual(runner.calls[0]["context_q_lens"], (3, 1))
        self.assertEqual(runner.calls[0]["decode_rows"], (1,))
        self.assertEqual(
            committed,
            [("prefill", "prefill"), ("decode", prepared_decode)],
        )
        self.assertEqual(engine.metrics.mixed_batch_fallbacks, 0)
        self.assertEqual(engine.metrics.mixed_batch_opportunities, 1)
        self.assertEqual(engine.metrics.decode_rows, 1)
        self.assertEqual(engine.metrics.max_decode_batch_rows, 1)
        self.assertEqual(engine.metrics.max_decode_model_batch_rows, 1)
        self.assertEqual(engine.metrics.batched_decode_model_calls, 0)

    def test_missing_token_pool_layer_coverage_falls_back_before_forward(self):
        config = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=8,
        )

        class Model:
            wkvm_no_hf_transformer_forward = True
            device = torch.device("cpu")

            def parameters(self):
                yield torch.zeros(1)

        class Cache:
            def state_bytes(self):
                return 0

        class Runner:
            model = Model()

            def __init__(self):
                self.model_calls = 0

            def build_cache(self, _slots):
                return Cache()

            def mixed_batch_step(self, *_args, **_kwargs):
                self.model_calls += 1
                raise AssertionError("coverage fallback must precede model forward")

        class DecodeContext:
            def covered_decode_layer_types(self):
                return frozenset()

        engine = GemmaNativeEngine(
            model=Model(),
            config=config,
            num_slots=2,
            enable_mixed_batch=True,
            enable_token_pool_metadata=False,
        )
        runner = Runner()
        engine.runner = runner  # type: ignore[assignment]
        engine.enable_token_pool_metadata = True
        engine.enable_token_pool_attention = True
        engine._token_kv_pool = SimpleNamespace(layer_specs={0: object()})
        engine._token_pool_decode_backend = object()  # type: ignore[assignment]
        prepared_decode = (object(),)
        discarded = []
        engine._token_pool_prepare_authoritative_prefill = (  # type: ignore[method-assign]
            lambda *_args, **_kwargs: None
        )
        engine._token_pool_prepare_decode_model_batch = (  # type: ignore[method-assign]
            lambda _reqs: prepared_decode
        )
        engine._token_pool_decode_context = (  # type: ignore[method-assign]
            lambda _prepared: DecodeContext()
        )
        engine._token_pool_layer_plan = (  # type: ignore[method-assign]
            lambda: SimpleNamespace(
                layer_type_by_layer_id={0: "sliding_attention"}
            )
        )
        engine._token_pool_discard_decode_reservations = (  # type: ignore[method-assign]
            lambda prepared: discarded.append(prepared)
        )
        prefill = Request([1, 2, 3], max_new_tokens=1, req_id="prefill")
        decode = Request([4], max_new_tokens=2, req_id="decode")
        decode.num_computed_tokens = 1
        decode.output_token_ids = [9]
        decode.slots = {"gemma_routed_span": 1}
        engine._caches[decode.req_id] = Cache()  # type: ignore[assignment]

        actual = engine._execute_mixed_batch(
            [(prefill, 3, True), (decode, 1, False)]
        )

        self.assertIsNone(actual)
        self.assertEqual(runner.model_calls, 0)
        self.assertEqual(discarded, [prepared_decode])
        self.assertNotIn(prefill.req_id, engine._caches)
        self.assertEqual(engine.metrics.cache_builds, 0)

    def test_mixed_forward_error_rolls_back_initial_cache_build(self):
        config = gemma4_e4b_routed_span_config(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=("sliding_attention",),
            sliding_window=8,
        )

        class Model:
            wkvm_no_hf_transformer_forward = True
            device = torch.device("cpu")

            def parameters(self):
                yield torch.zeros(1)

        class Cache:
            def state_bytes(self):
                return 0

        class Runner:
            model = Model()

            def build_cache(self, _slots):
                return Cache()

            def mixed_batch_step(self, *_args, **_kwargs):
                raise RuntimeError("synthetic mixed forward failure")

        engine = GemmaNativeEngine(
            model=Model(),
            config=config,
            num_slots=2,
            enable_mixed_batch=True,
            enable_token_pool_metadata=False,
        )
        engine.runner = Runner()  # type: ignore[assignment]
        prefill = Request([1, 2, 3], max_new_tokens=1, req_id="prefill")
        decode = Request([4], max_new_tokens=2, req_id="decode")
        decode.num_computed_tokens = 1
        decode.output_token_ids = [9]
        decode.slots = {"gemma_routed_span": 1}
        engine._caches[decode.req_id] = Cache()  # type: ignore[assignment]

        with self.assertRaisesRegex(RuntimeError, "synthetic mixed forward failure"):
            engine._execute_mixed_batch(
                [(prefill, 3, True), (decode, 1, False)]
            )

        self.assertNotIn(prefill.req_id, engine._caches)
        self.assertIn(decode.req_id, engine._caches)
        self.assertEqual(engine.metrics.cache_builds, 0)


if __name__ == "__main__":
    unittest.main()
