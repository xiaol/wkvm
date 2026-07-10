import unittest
from collections import UserDict


def _tiny_config():
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    return Gemma4TextConfig(
        vocab_size=64,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        hidden_size_per_layer_input=4,
        vocab_size_per_layer_input=64,
        sliding_window=8,
        layer_types=["sliding_attention", "sliding_attention"],
        num_kv_shared_layers=0,
        attention_dropout=0.0,
        attention_bias=False,
        global_head_dim=4,
    )


def _tiny_shared_config():
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    return Gemma4TextConfig(
        vocab_size=64,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        hidden_size_per_layer_input=4,
        vocab_size_per_layer_input=64,
        sliding_window=8,
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
    )


def _causal_mask(batch: int, seq_len: int, *, dtype, device):
    import torch

    mask = torch.zeros(batch, 1, seq_len, seq_len, dtype=dtype, device=device)
    blocked = torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
        diagonal=1,
    )
    return mask.masked_fill(blocked.view(1, 1, seq_len, seq_len), torch.finfo(dtype).min)


def _checkpoint_layout_state_dict(hf_model):
    state = {}
    for key, value in hf_model.state_dict().items():
        if key.startswith("model."):
            key = "model.language_model" + key[len("model") :]
        state[key] = value.detach().clone()
    return state


def _token_pool_backend_decode(
    native_forward,
    attn,
    query_states,
    *,
    decode_metadata,
    paged_decode_metadata=None,
    token_kv_pool,
    layer_idx: int,
    token_pool_plan=None,
    current_key_states=None,
    current_value_states=None,
):
    from wkvm.runner.gemma_token_pool import build_token_pool_attention_call

    attention_call = build_token_pool_attention_call(
        token_pool_plan=token_pool_plan,
        decode_metadata=decode_metadata,
        paged_decode_metadata=paged_decode_metadata,
        token_kv_pool=token_kv_pool,
        layer_idx=layer_idx,
        attention_mask_present=False,
        query_seq_len=query_states.shape[2],
    ).with_current_kv(
        current_key_states,
        current_value_states,
    )
    result = native_forward._token_pool_attention_backend().decode_call(
        attn,
        query_states,
        attention_call=attention_call,
        timing_enabled=False,
    )
    return result.output, result.weights


class TestNativeGemma4TextDecoderLayer(unittest.TestCase):
    def test_decoder_layer_exposes_native_attention_boundary(self) -> None:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer
        from wkvm.runner.gemma_native_forward import (
            NativeGemma4Attention,
            NativeGemma4AttentionBackend,
            NativeGemma4TextDecoderLayer,
        )

        cfg = _tiny_config()
        hf_layer = Gemma4TextDecoderLayer(cfg, layer_idx=0).eval()
        native_layer = NativeGemma4TextDecoderLayer(hf_layer)

        self.assertIsInstance(native_layer.self_attn, NativeGemma4Attention)
        self.assertIsInstance(
            native_layer.self_attn.attention_backend,
            NativeGemma4AttentionBackend,
        )
        self.assertTrue(hasattr(native_layer.self_attn, "shared_kv_state"))
        self.assertIs(native_layer.attn_meta, native_layer.self_attn.attn_meta)
        self.assertIs(native_layer.q_proj, native_layer.self_attn.q_proj)
        self.assertIs(native_layer.o_proj, native_layer.self_attn.o_proj)
        self.assertEqual(native_layer.layer_type, native_layer.attn_meta.layer_type)
        self.assertIsNone(native_layer.attn_meta.kv_shared_layer_index)

    def test_shared_kv_source_layer_index_is_model_side_metadata(self) -> None:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer
        from wkvm.runner.gemma_native_forward import (
            NativeGemma4TextDecoderLayer,
            _CheckpointGemma4Attention,
        )

        cfg = _tiny_shared_config()
        source_layer = NativeGemma4TextDecoderLayer(
            Gemma4TextDecoderLayer(cfg, layer_idx=0).eval()
        )
        shared_sliding_hf = Gemma4TextDecoderLayer(cfg, layer_idx=2).eval()
        shared_sliding = NativeGemma4TextDecoderLayer(shared_sliding_hf)
        shared_full_hf = Gemma4TextDecoderLayer(cfg, layer_idx=3).eval()
        shared_full = NativeGemma4TextDecoderLayer(shared_full_hf)

        self.assertFalse(source_layer.attn_meta.is_kv_shared_layer)
        self.assertTrue(source_layer.attn_meta.store_full_length_kv)
        self.assertIsNone(source_layer.attn_meta.kv_shared_layer_index)

        self.assertTrue(shared_sliding.attn_meta.is_kv_shared_layer)
        self.assertFalse(shared_sliding.attn_meta.store_full_length_kv)
        self.assertEqual(shared_sliding.attn_meta.kv_shared_layer_index, 0)

        self.assertTrue(shared_full.attn_meta.is_kv_shared_layer)
        self.assertEqual(shared_full.attn_meta.kv_shared_layer_index, 1)

        checkpoint_attn = _CheckpointGemma4Attention(
            cfg,
            3,
            shared_full_hf.state_dict(),
            "self_attn",
        )
        self.assertTrue(checkpoint_attn.is_kv_shared_layer)
        self.assertEqual(checkpoint_attn.kv_shared_layer_index, 1)

    def test_attention_owns_shared_kv_state_handoff(self) -> None:
        import torch
        from wkvm.runner.gemma_native_forward import (
            NativeGemma4SharedKVState,
            _NativeAttentionMeta,
        )

        shared_kv_state = NativeGemma4SharedKVState()
        shared_meta = _NativeAttentionMeta(
            head_dim=4,
            num_key_value_groups=1,
            attention_dropout=0.0,
            training=False,
            scaling=1.0,
            is_kv_shared_layer=True,
            layer_type="sliding_attention",
            kv_shared_layer_index=0,
            store_full_length_kv=False,
        )
        key_states = torch.randn(1, 1, 3, 4)
        value_states = torch.randn(1, 1, 3, 4)
        shared_kv_states = UserDict(
            {"sliding_attention": (key_states, value_states)}
        )

        loaded = shared_kv_state.load_shared_kv(
            shared_meta,
            shared_kv_states,
            query_device=key_states.device,
            timing_enabled=False,
        )

        self.assertIsNotNone(loaded)
        self.assertIs(loaded[0], key_states)
        self.assertIs(loaded[1], value_states)
        with self.assertRaises(KeyError):
            shared_kv_state.load_shared_kv(
                shared_meta,
                UserDict(),
                query_device=key_states.device,
                timing_enabled=False,
            )

        store_meta = _NativeAttentionMeta(
            head_dim=4,
            num_key_value_groups=1,
            attention_dropout=0.0,
            training=False,
            scaling=1.0,
            is_kv_shared_layer=False,
            layer_type="sliding_attention",
            kv_shared_layer_index=None,
            store_full_length_kv=True,
        )
        stored = UserDict()
        shared_kv_state.store_shared_kv(store_meta, stored, key_states, value_states)

        self.assertIs(stored["sliding_attention"][0], key_states)
        self.assertIs(stored["sliding_attention"][1], value_states)

    def test_attention_backend_resolves_shared_kv_source_layer(self) -> None:
        from types import SimpleNamespace
        from wkvm.runner.gemma_native_forward import (
            NativeGemma4AttentionBackend,
            _NativeAttentionMeta,
        )

        events = []

        class QueryStates:
            shape = (1, 2, 1, 4)

        class DecodeContext:
            def attention_plan_for_layer(
                self,
                layer_idx,
                layer_type,
                *,
                attention_mask_present: bool = False,
                query_seq_len=None,
            ):
                events.append(
                    (layer_idx, layer_type, attention_mask_present, query_seq_len)
                )
                return SimpleNamespace(
                    attention_kwargs=lambda: {
                        "decode_metadata": "metadata",
                        "paged_decode_metadata": None,
                        "token_kv_pool": "pool",
                        "layer_idx": layer_idx,
                    },
                    decode_attention_enabled=lambda: True,
                )

        meta = _NativeAttentionMeta(
            head_dim=4,
            num_key_value_groups=1,
            attention_dropout=0.0,
            training=False,
            scaling=1.0,
            is_kv_shared_layer=True,
            layer_type="sliding_attention",
            kv_shared_layer_index=0,
            store_full_length_kv=False,
        )
        backend = NativeGemma4AttentionBackend("manual_gqa")

        attention_call = backend.resolve_attention_call(
            meta,
            layer_idx=2,
            query_states=QueryStates(),
            attention_mask=None,
            wkvm_token_pool_decode=DecodeContext(),
            timing_enabled=False,
        )
        binding = attention_call.bind_layer_kv(
            "key",
            "value",
            has_past_key_values=True,
            is_kv_shared_layer=True,
        )

        self.assertEqual(events, [(0, "sliding_attention", False, 1)])
        self.assertEqual(attention_call.backend_decode_kwargs()["layer_idx"], 0)
        self.assertIsNone(binding.attention_call.current_kv_for_backend()[0])
        self.assertIsNone(binding.attention_call.current_kv_for_backend()[1])
        self.assertFalse(binding.should_update_dense_cache)

    def test_prefill_layer_matches_hf_decoder_layer(self) -> None:
        import torch
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextDecoderLayer,
            Gemma4TextRotaryEmbedding,
        )
        from wkvm.runner.gemma_native_forward import NativeGemma4TextDecoderLayer

        torch.manual_seed(7)
        cfg = _tiny_config()
        hf_layer = Gemma4TextDecoderLayer(cfg, layer_idx=0).eval()
        native_layer = NativeGemma4TextDecoderLayer(hf_layer)
        rotary = Gemma4TextRotaryEmbedding(cfg)

        hidden = torch.randn(2, 5, cfg.hidden_size)
        per_layer_input = torch.randn(2, 5, cfg.hidden_size_per_layer_input)
        position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)
        position_embeddings = rotary(hidden, position_ids, "sliding_attention")
        mask = _causal_mask(2, 5, dtype=hidden.dtype, device=hidden.device)

        with torch.inference_mode():
            expected = hf_layer(
                hidden,
                per_layer_input,
                shared_kv_states=UserDict(),
                position_embeddings=position_embeddings,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_values=None,
            )
            actual = native_layer(
                hidden,
                per_layer_input,
                shared_kv_states=UserDict(),
                position_embeddings=position_embeddings,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_values=None,
            )

        self.assertLess((expected - actual).abs().max().item(), 2e-6)

    def test_decode_layer_with_wkvm_cache_matches_hf_decoder_layer(self) -> None:
        import torch
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextDecoderLayer,
            Gemma4TextRotaryEmbedding,
        )
        from wkvm.models.gemma import GemmaRoutedSpanConfig
        from wkvm.runner.gemma_native_forward import NativeGemma4TextDecoderLayer
        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        torch.manual_seed(11)
        cfg = _tiny_config()
        hf_layer = Gemma4TextDecoderLayer(cfg, layer_idx=0).eval()
        native_layer = NativeGemma4TextDecoderLayer(hf_layer)
        rotary = Gemma4TextRotaryEmbedding(cfg)
        native_cfg = GemmaRoutedSpanConfig(
            num_hidden_layers=cfg.num_hidden_layers,
            num_kv_shared_layers=cfg.num_kv_shared_layers,
            layer_types=tuple(cfg.layer_types),
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            sliding_window=cfg.sliding_window,
        )
        hf_cache = NativeGemmaRoutedCache(cfg, native_cfg)
        native_cache = NativeGemmaRoutedCache(cfg, native_cfg)

        prefill_hidden = torch.randn(1, 3, cfg.hidden_size)
        prefill_ple = torch.randn(1, 3, cfg.hidden_size_per_layer_input)
        prefill_pos = torch.arange(3).unsqueeze(0)
        prefill_pos_emb = rotary(prefill_hidden, prefill_pos, "sliding_attention")
        prefill_mask = _causal_mask(1, 3, dtype=prefill_hidden.dtype, device=prefill_hidden.device)

        decode_hidden = torch.randn(1, 1, cfg.hidden_size)
        decode_ple = torch.randn(1, 1, cfg.hidden_size_per_layer_input)
        decode_pos = torch.tensor([[3]])
        decode_pos_emb = rotary(decode_hidden, decode_pos, "sliding_attention")

        with torch.inference_mode():
            hf_prefill = hf_layer(
                prefill_hidden,
                prefill_ple,
                shared_kv_states=UserDict(),
                position_embeddings=prefill_pos_emb,
                attention_mask=prefill_mask,
                position_ids=prefill_pos,
                past_key_values=hf_cache,
            )
            native_prefill = native_layer(
                prefill_hidden,
                prefill_ple,
                shared_kv_states=UserDict(),
                position_embeddings=prefill_pos_emb,
                attention_mask=prefill_mask,
                position_ids=prefill_pos,
                past_key_values=native_cache,
            )
            hf_decode = hf_layer(
                decode_hidden,
                decode_ple,
                shared_kv_states=UserDict(),
                position_embeddings=decode_pos_emb,
                attention_mask=None,
                position_ids=decode_pos,
                past_key_values=hf_cache,
            )
            native_decode = native_layer(
                decode_hidden,
                decode_ple,
                shared_kv_states=UserDict(),
                position_embeddings=decode_pos_emb,
                attention_mask=None,
                position_ids=decode_pos,
                past_key_values=native_cache,
            )

        self.assertLess((hf_prefill - native_prefill).abs().max().item(), 1e-6)
        self.assertLess((hf_decode - native_decode).abs().max().item(), 1e-6)
        self.assertEqual(hf_cache.layers[0].keys.shape, native_cache.layers[0].keys.shape)
        self.assertLess(
            (hf_cache.layers[0].keys - native_cache.layers[0].keys).abs().max().item(),
            1e-6,
        )

    def test_manual_gqa_backend_matches_hf_decoder_layer(self) -> None:
        import torch
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextDecoderLayer,
            Gemma4TextRotaryEmbedding,
        )
        from wkvm.models.gemma import GemmaRoutedSpanConfig
        from wkvm.runner.gemma_native_forward import NativeGemma4TextDecoderLayer
        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        torch.manual_seed(17)
        cfg = _tiny_config()
        hf_layer = Gemma4TextDecoderLayer(cfg, layer_idx=0).eval()
        native_layer = NativeGemma4TextDecoderLayer(
            hf_layer,
            native_attention_backend="manual_gqa",
        )
        rotary = Gemma4TextRotaryEmbedding(cfg)
        native_cfg = GemmaRoutedSpanConfig(
            num_hidden_layers=cfg.num_hidden_layers,
            num_kv_shared_layers=cfg.num_kv_shared_layers,
            layer_types=tuple(cfg.layer_types),
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            sliding_window=cfg.sliding_window,
        )
        hf_cache = NativeGemmaRoutedCache(cfg, native_cfg)
        native_cache = NativeGemmaRoutedCache(cfg, native_cfg)

        prefill_hidden = torch.randn(1, 4, cfg.hidden_size)
        prefill_ple = torch.randn(1, 4, cfg.hidden_size_per_layer_input)
        prefill_pos = torch.arange(4).unsqueeze(0)
        prefill_pos_emb = rotary(prefill_hidden, prefill_pos, "sliding_attention")
        prefill_mask = _causal_mask(1, 4, dtype=prefill_hidden.dtype, device=prefill_hidden.device)
        decode_hidden = torch.randn(1, 1, cfg.hidden_size)
        decode_ple = torch.randn(1, 1, cfg.hidden_size_per_layer_input)
        decode_pos = torch.tensor([[4]])
        decode_pos_emb = rotary(decode_hidden, decode_pos, "sliding_attention")

        with torch.inference_mode():
            hf_prefill = hf_layer(
                prefill_hidden,
                prefill_ple,
                shared_kv_states=UserDict(),
                position_embeddings=prefill_pos_emb,
                attention_mask=prefill_mask,
                position_ids=prefill_pos,
                past_key_values=hf_cache,
            )
            native_prefill = native_layer(
                prefill_hidden,
                prefill_ple,
                shared_kv_states=UserDict(),
                position_embeddings=prefill_pos_emb,
                attention_mask=prefill_mask,
                position_ids=prefill_pos,
                past_key_values=native_cache,
            )
            hf_decode = hf_layer(
                decode_hidden,
                decode_ple,
                shared_kv_states=UserDict(),
                position_embeddings=decode_pos_emb,
                attention_mask=None,
                position_ids=decode_pos,
                past_key_values=hf_cache,
            )
            native_decode = native_layer(
                decode_hidden,
                decode_ple,
                shared_kv_states=UserDict(),
                position_embeddings=decode_pos_emb,
                attention_mask=None,
                position_ids=decode_pos,
                past_key_values=native_cache,
            )

        self.assertLess((hf_prefill - native_prefill).abs().max().item(), 2e-6)
        self.assertLess((hf_decode - native_decode).abs().max().item(), 2e-6)

    def test_triton_dense_gqa_backend_matches_hf_decoder_layer_on_cpu_fallback(self) -> None:
        import torch
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextDecoderLayer,
            Gemma4TextRotaryEmbedding,
        )
        from wkvm.models.gemma import GemmaRoutedSpanConfig
        from wkvm.runner.gemma_native_forward import NativeGemma4TextDecoderLayer
        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        torch.manual_seed(23)
        cfg = _tiny_config()
        hf_layer = Gemma4TextDecoderLayer(cfg, layer_idx=0).eval()
        native_layer = NativeGemma4TextDecoderLayer(
            hf_layer,
            native_attention_backend="triton_dense_gqa",
        )
        rotary = Gemma4TextRotaryEmbedding(cfg)
        native_cfg = GemmaRoutedSpanConfig(
            num_hidden_layers=cfg.num_hidden_layers,
            num_kv_shared_layers=cfg.num_kv_shared_layers,
            layer_types=tuple(cfg.layer_types),
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            sliding_window=cfg.sliding_window,
        )
        hf_cache = NativeGemmaRoutedCache(cfg, native_cfg)
        native_cache = NativeGemmaRoutedCache(cfg, native_cfg)

        prefill_hidden = torch.randn(1, 4, cfg.hidden_size)
        prefill_ple = torch.randn(1, 4, cfg.hidden_size_per_layer_input)
        prefill_pos = torch.arange(4).unsqueeze(0)
        prefill_pos_emb = rotary(prefill_hidden, prefill_pos, "sliding_attention")
        prefill_mask = _causal_mask(1, 4, dtype=prefill_hidden.dtype, device=prefill_hidden.device)
        decode_hidden = torch.randn(1, 1, cfg.hidden_size)
        decode_ple = torch.randn(1, 1, cfg.hidden_size_per_layer_input)
        decode_pos = torch.tensor([[4]])
        decode_pos_emb = rotary(decode_hidden, decode_pos, "sliding_attention")

        with torch.inference_mode():
            hf_layer(
                prefill_hidden,
                prefill_ple,
                shared_kv_states=UserDict(),
                position_embeddings=prefill_pos_emb,
                attention_mask=prefill_mask,
                position_ids=prefill_pos,
                past_key_values=hf_cache,
            )
            native_layer(
                prefill_hidden,
                prefill_ple,
                shared_kv_states=UserDict(),
                position_embeddings=prefill_pos_emb,
                attention_mask=prefill_mask,
                position_ids=prefill_pos,
                past_key_values=native_cache,
            )
            hf_decode = hf_layer(
                decode_hidden,
                decode_ple,
                shared_kv_states=UserDict(),
                position_embeddings=decode_pos_emb,
                attention_mask=None,
                position_ids=decode_pos,
                past_key_values=hf_cache,
            )
            native_decode = native_layer(
                decode_hidden,
                decode_ple,
                shared_kv_states=UserDict(),
                position_embeddings=decode_pos_emb,
                attention_mask=None,
                position_ids=decode_pos,
                past_key_values=native_cache,
            )

        self.assertLess((hf_decode - native_decode).abs().max().item(), 2e-6)

    def test_token_pool_attention_matches_ragged_manual_gqa(self) -> None:
        import torch
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer
        from wkvm.runner.gemma_native_forward import (
            NativeGemma4TextDecoderLayer,
            _attention_forward,
            _attention_forward_manual_gqa,
        )
        from wkvm.runner.gemma_token_pool import (
            TokenKVLayerSpec,
            TokenKVPool,
            build_decode_metadata_from_token_slot_rows,
        )

        torch.manual_seed(19)
        cfg = _tiny_config()
        hf_layer = Gemma4TextDecoderLayer(cfg, layer_idx=0).eval()
        native_layer = NativeGemma4TextDecoderLayer(
            hf_layer,
            native_attention_backend="manual_gqa",
        )
        pool = TokenKVPool(
            capacity=5,
            layer_specs=[
                TokenKVLayerSpec(
                    layer_id=0,
                    num_kv_heads=cfg.num_key_value_heads,
                    head_dim=cfg.head_dim,
                    dtype=torch.float32,
                )
            ],
            dtype=torch.float32,
        )
        slots = pool.alloc_slots(5)
        keys = torch.randn(5, cfg.num_key_value_heads, cfg.head_dim)
        values = torch.randn(5, cfg.num_key_value_heads, cfg.head_dim)
        pool.set_kv(0, slots, keys, values)
        query = torch.randn(2, cfg.num_attention_heads, 1, cfg.head_dim)
        metadata = build_decode_metadata_from_token_slot_rows(
            [[2, 0, 1], [4, 3]],
            req_slots=[0, 1],
            out_cache_loc=[2, 4],
        )

        with torch.inference_mode():
            actual, actual_weights = _attention_forward(
                native_layer.attn_meta,
                query,
                keys[:1].permute(1, 0, 2).unsqueeze(0),
                values[:1].permute(1, 0, 2).unsqueeze(0),
                None,
                backend="manual_gqa",
                decode_metadata=metadata,
                token_kv_pool=pool,
                layer_idx=0,
            )
            expected_rows = []
            for row_indices, row in (([2, 0, 1], 0), ([4, 3], 1)):
                row_keys = keys[row_indices].permute(1, 0, 2).unsqueeze(0)
                row_values = values[row_indices].permute(1, 0, 2).unsqueeze(0)
                expected, _ = _attention_forward_manual_gqa(
                    native_layer.attn_meta,
                    query[row : row + 1],
                    row_keys,
                    row_values,
                    None,
                )
                expected_rows.append(expected)
            expected = torch.cat(expected_rows, dim=0)

        self.assertIsNone(actual_weights)
        self.assertLess((expected - actual).abs().max().item(), 1e-6)

    def test_attention_forward_respects_disabled_token_pool_plan(self) -> None:
        import torch
        import wkvm.runner.gemma_native_forward as native_forward
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer
        from wkvm.runner.gemma_native_forward import (
            NativeGemma4TextDecoderLayer,
            _attention_forward,
            _attention_forward_manual_gqa,
        )
        from wkvm.runner.gemma_token_pool import (
            TokenKVLayerSpec,
            TokenKVPool,
            TokenPoolAttentionPlan,
            build_decode_metadata_from_token_slot_rows,
        )

        torch.manual_seed(195)
        cfg = _tiny_config()
        hf_layer = Gemma4TextDecoderLayer(cfg, layer_idx=0).eval()
        native_layer = NativeGemma4TextDecoderLayer(
            hf_layer,
            native_attention_backend="manual_gqa",
        )
        pool = TokenKVPool(
            capacity=3,
            layer_specs=[
                TokenKVLayerSpec(
                    layer_id=0,
                    num_kv_heads=cfg.num_key_value_heads,
                    head_dim=cfg.head_dim,
                    dtype=torch.float32,
                )
            ],
            dtype=torch.float32,
        )
        slots = pool.alloc_slots(3)
        pool.set_kv(
            0,
            slots,
            torch.randn(3, cfg.num_key_value_heads, cfg.head_dim),
            torch.randn(3, cfg.num_key_value_heads, cfg.head_dim),
        )
        metadata = build_decode_metadata_from_token_slot_rows(
            [[0, 1, 2]],
            out_cache_loc=[2],
        )
        plan = TokenPoolAttentionPlan(
            layer_idx=0,
            metadata=metadata,
            paged_metadata=None,
            kv_pool=pool,
            use_decode_attention=False,
        )
        query = torch.randn(1, cfg.num_attention_heads, 1, cfg.head_dim)
        dense_keys = torch.randn(1, cfg.num_key_value_heads, 2, cfg.head_dim)
        dense_values = torch.randn(1, cfg.num_key_value_heads, 2, cfg.head_dim)
        old_backend = native_forward._TOKEN_POOL_ATTENTION_BACKEND
        backend_seen_enabled = []

        class Backend:
            def try_decode_call(self, actual_attn, actual_query_states, **kwargs):
                del actual_attn, actual_query_states
                backend_seen_enabled.append(
                    kwargs["attention_call"].decode_attention_enabled
                )
                return None

            def decode_call(self, *args, **kwargs):
                raise AssertionError("native forward should call try_decode_call")

        try:
            native_forward._TOKEN_POOL_ATTENTION_BACKEND = Backend()
            with torch.inference_mode():
                expected, expected_weights = _attention_forward_manual_gqa(
                    native_layer.attn_meta,
                    query,
                    dense_keys,
                    dense_values,
                    None,
                )
                actual, actual_weights = _attention_forward(
                    native_layer.attn_meta,
                    query,
                    dense_keys,
                    dense_values,
                    None,
                    backend="manual_gqa",
                    token_pool_plan=plan,
                )
        finally:
            native_forward._TOKEN_POOL_ATTENTION_BACKEND = old_backend

        self.assertEqual(backend_seen_enabled, [False])
        self.assertIsNotNone(actual_weights)
        self.assertLess((expected - actual).abs().max().item(), 1e-6)
        self.assertLess((expected_weights - actual_weights).abs().max().item(), 1e-6)

    def test_attention_forward_uses_plan_attention_kwargs_and_enabled_hook(self) -> None:
        from types import SimpleNamespace
        import wkvm.runner.gemma_native_forward as native_forward

        old_backend = native_forward._TOKEN_POOL_ATTENTION_BACKEND
        events = []
        calls = {}

        class FakeQuery:
            shape = (1, 8, 1, 512)

        class MethodOnlyPlan:
            def attention_kwargs(self):
                events.append("kwargs")
                return {
                    "decode_metadata": "owned_metadata",
                    "paged_decode_metadata": "owned_paged_metadata",
                    "token_kv_pool": "owned_pool",
                    "layer_idx": 17,
                }

            def decode_attention_enabled(self):
                events.append("enabled")
                return True

        class Backend:
            def try_decode_call(self, actual_attn, actual_query_states, **kwargs):
                events.append("dispatch")
                calls["attn"] = actual_attn
                calls["query_states"] = actual_query_states
                calls.update(kwargs["attention_call"].backend_decode_kwargs())
                return SimpleNamespace(output="token_pool", weights=None)

            def decode_call(self, *args, **kwargs):
                raise AssertionError("native forward should call try_decode_call")

        try:
            native_forward._TOKEN_POOL_ATTENTION_BACKEND = Backend()
            actual = native_forward._attention_forward(
                SimpleNamespace(num_key_value_groups=4, scaling=1.0),
                FakeQuery(),
                "dense_key_states",
                "dense_value_states",
                None,
                backend="manual_gqa",
                token_pool_plan=MethodOnlyPlan(),
                current_key_states="current_key_states",
                current_value_states="current_value_states",
            )
        finally:
            native_forward._TOKEN_POOL_ATTENTION_BACKEND = old_backend

        self.assertEqual(actual, ("token_pool", None))
        self.assertEqual(events, ["kwargs", "enabled", "dispatch"])
        self.assertEqual(calls["decode_metadata"], "owned_metadata")
        self.assertEqual(calls["paged_decode_metadata"], "owned_paged_metadata")
        self.assertEqual(calls["token_kv_pool"], "owned_pool")
        self.assertEqual(calls["layer_idx"], 17)
        self.assertIsInstance(calls["token_pool_plan"], MethodOnlyPlan)
        self.assertEqual(calls["current_key_states"], "current_key_states")
        self.assertEqual(calls["current_value_states"], "current_value_states")

    def test_token_pool_attention_adapter_delegates_to_backend_call(self) -> None:
        from types import SimpleNamespace
        import wkvm.runner.gemma_native_forward as native_forward

        old_backend = native_forward._TOKEN_POOL_ATTENTION_BACKEND
        calls = {}

        class FakeQuery:
            shape = (1, 8, 1, 512)

        class Backend:
            def decode_call(self, actual_attn, actual_query_states, **kwargs):
                calls["attn"] = actual_attn
                calls["query_states"] = actual_query_states
                calls.update(kwargs["attention_call"].backend_decode_kwargs())
                return SimpleNamespace(output="token_pool", weights="weights")

        attn = SimpleNamespace(num_key_value_groups=4, scaling=1.0)
        query_states = FakeQuery()
        try:
            native_forward._TOKEN_POOL_ATTENTION_BACKEND = Backend()
            actual = native_forward._attention_forward_token_pool_gqa(
                attn,
                query_states,
                decode_metadata="metadata",
                paged_decode_metadata="paged_metadata",
                token_kv_pool="pool",
                layer_idx=5,
                current_key_states="key",
                current_value_states="value",
            )
        finally:
            native_forward._TOKEN_POOL_ATTENTION_BACKEND = old_backend

        self.assertEqual(actual, ("token_pool", "weights"))
        self.assertIs(calls["attn"], attn)
        self.assertIs(calls["query_states"], query_states)
        self.assertEqual(calls["decode_metadata"], "metadata")
        self.assertEqual(calls["paged_decode_metadata"], "paged_metadata")
        self.assertEqual(calls["token_kv_pool"], "pool")
        self.assertEqual(calls["layer_idx"], 5)
        self.assertIsNone(calls["token_pool_plan"])
        self.assertEqual(calls["current_key_states"], "key")
        self.assertEqual(calls["current_value_states"], "value")

    def test_token_pool_triton_stats_account_recoverable_fallback(self) -> None:
        import os
        import sys
        from types import ModuleType, SimpleNamespace
        import wkvm.runner.gemma_native_forward as native_forward

        native_forward.reset_token_pool_triton_stats(clear_disabled_shapes=True)
        module_name = "wkvm.runner.gemma_token_pool_triton"
        old_module = sys.modules.get(module_name)
        old_flag = os.environ.get("WKVM_ENABLE_TOKEN_POOL_TRITON")
        old_disable = os.environ.get("WKVM_DISABLE_TOKEN_POOL_TRITON")
        old_reference = native_forward._attention_forward_token_pool_gqa_reference
        fake_module = ModuleType(module_name)

        def fail_decode(*args, **kwargs):
            raise RuntimeError("out of resource: shared memory")

        fake_module.token_pool_gqa_decode = fail_decode
        sys.modules[module_name] = fake_module
        os.environ["WKVM_ENABLE_TOKEN_POOL_TRITON"] = "1"
        os.environ.pop("WKVM_DISABLE_TOKEN_POOL_TRITON", None)
        native_forward._attention_forward_token_pool_gqa_reference = (
            lambda *args, **kwargs: ("reference", None)
        )

        class FakeQuery:
            shape = (1, 8, 1, 512)
            dtype = "bfloat16"
            device = "cuda:0"
            is_cuda = True

        class FakePool:
            def get_kv_buffer(self, layer_idx):
                return "key_buffer", "value_buffer"

        try:
            actual = _token_pool_backend_decode(
                native_forward,
                SimpleNamespace(num_key_value_groups=4, scaling=1.0),
                FakeQuery(),
                decode_metadata=SimpleNamespace(kv_indptr="indptr", kv_indices="indices"),
                token_kv_pool=FakePool(),
                layer_idx=0,
            )
            self.assertEqual(actual, ("reference", None))
            second = _token_pool_backend_decode(
                native_forward,
                SimpleNamespace(num_key_value_groups=4, scaling=1.0),
                FakeQuery(),
                decode_metadata=SimpleNamespace(kv_indptr="indptr", kv_indices="indices"),
                token_kv_pool=FakePool(),
                layer_idx=0,
            )
            self.assertEqual(second, ("reference", None))
            stats = native_forward.token_pool_triton_stats()
        finally:
            native_forward._attention_forward_token_pool_gqa_reference = old_reference
            if old_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = old_module
            if old_flag is None:
                os.environ.pop("WKVM_ENABLE_TOKEN_POOL_TRITON", None)
            else:
                os.environ["WKVM_ENABLE_TOKEN_POOL_TRITON"] = old_flag
            if old_disable is None:
                os.environ.pop("WKVM_DISABLE_TOKEN_POOL_TRITON", None)
            else:
                os.environ["WKVM_DISABLE_TOKEN_POOL_TRITON"] = old_disable
            native_forward.reset_token_pool_triton_stats(clear_disabled_shapes=True)

        self.assertEqual(stats["calls"], 2)
        self.assertEqual(stats["env_enabled_calls"], 2)
        self.assertEqual(stats["attempts"], 1)
        self.assertEqual(stats["successes"], 0)
        self.assertEqual(stats["runtime_errors"], 1)
        self.assertEqual(stats["recoverable_runtime_fallbacks"], 1)
        self.assertEqual(stats["disabled_shape_skips"], 1)
        self.assertEqual(stats["disabled_shape_count"], 1)
        self.assertEqual(
            stats["fallback_reasons"],
            {"recoverable_runtime_error": 1, "disabled_shape": 1},
        )

    def test_token_pool_attention_dispatch_stores_current_kv_before_attention(self) -> None:
        import os
        from types import SimpleNamespace
        import torch
        import wkvm.runner.gemma_native_forward as native_forward

        old_disable = os.environ.get("WKVM_DISABLE_TOKEN_POOL_TRITON")
        old_reference = native_forward._attention_forward_token_pool_gqa_reference
        native_forward.reset_token_pool_triton_stats(clear_disabled_shapes=True)
        events: list[str] = []
        query_states = torch.randn(1, 1, 1, 2)
        key_states = torch.randn(1, 1, 1, 2)
        value_states = torch.randn(1, 1, 1, 2)

        class Plan:
            metadata = SimpleNamespace(
                kv_indptr=torch.tensor([0, 1], dtype=torch.int32),
                kv_indices=torch.tensor([0], dtype=torch.int32),
                out_cache_loc=torch.tensor([0], dtype=torch.int32),
                out_cache_loc_long=torch.tensor([0], dtype=torch.long),
            )
            paged_metadata = None
            kv_pool = object()
            layer_idx = 0
            use_decode_attention = True

            def store_current_kv(self, key, value):
                events.append("store")
                self.stored_key = key
                self.stored_value = value
                return self.metadata.out_cache_loc

        plan = Plan()

        def reference(*_args, **_kwargs):
            events.append("reference")
            return torch.zeros_like(query_states), None

        try:
            os.environ["WKVM_DISABLE_TOKEN_POOL_TRITON"] = "1"
            native_forward._attention_forward_token_pool_gqa_reference = reference
            output, _weights = native_forward._attention_forward(
                SimpleNamespace(num_key_value_groups=1, scaling=1.0),
                query_states,
                key_states,
                value_states,
                None,
                backend="manual_gqa",
                token_pool_plan=plan,
                current_key_states=key_states,
                current_value_states=value_states,
            )
        finally:
            native_forward._attention_forward_token_pool_gqa_reference = old_reference
            if old_disable is None:
                os.environ.pop("WKVM_DISABLE_TOKEN_POOL_TRITON", None)
            else:
                os.environ["WKVM_DISABLE_TOKEN_POOL_TRITON"] = old_disable
            native_forward.reset_token_pool_triton_stats(clear_disabled_shapes=True)

        self.assertEqual(events, ["store", "reference"])
        self.assertIs(plan.stored_key, key_states)
        self.assertIs(plan.stored_value, value_states)
        self.assertTrue(torch.equal(output, torch.zeros_like(query_states)))

    def test_token_pool_triton_dispatch_uses_plan_owned_context(self) -> None:
        import os
        import sys
        from types import ModuleType, SimpleNamespace
        import wkvm.runner.gemma_native_forward as native_forward
        from wkvm.runner.gemma_token_pool import TokenPoolAttentionDispatchContext

        native_forward.reset_token_pool_triton_stats(clear_disabled_shapes=True)
        module_name = "wkvm.runner.gemma_token_pool_triton"
        old_module = sys.modules.get(module_name)
        old_enable = os.environ.get("WKVM_ENABLE_TOKEN_POOL_TRITON")
        old_disable = os.environ.get("WKVM_DISABLE_TOKEN_POOL_TRITON")
        fake_module = ModuleType(module_name)
        events = []
        calls = {}

        def decode(query, key_buffer, value_buffer, kv_indptr, kv_indices, **kwargs):
            events.append("decode")
            calls.update(
                {
                    "key_buffer": key_buffer,
                    "value_buffer": value_buffer,
                    "kv_indptr": kv_indptr,
                    "kv_indices": kv_indices,
                    "output": kwargs.get("output"),
                }
            )
            return "owned_triton"

        fake_module.token_pool_gqa_decode = decode
        sys.modules[module_name] = fake_module
        os.environ["WKVM_ENABLE_TOKEN_POOL_TRITON"] = "1"
        os.environ.pop("WKVM_DISABLE_TOKEN_POOL_TRITON", None)

        class FakeQuery:
            shape = (1, 8, 1, 512)
            dtype = "bfloat16"
            device = "cuda:0"
            is_cuda = True

        class FallbackPool:
            def get_kv_buffer(self, layer_idx):
                return "fallback_key", "fallback_value"

        class DispatchOwner:
            def kv_buffers_for_attention(self):
                events.append("kv_buffers")
                return "owned_key", "owned_value"

            def attention_output_buffer(self, **kwargs):
                events.append(("output", kwargs))
                return "owned_output"

        dispatch_owner = DispatchOwner()
        dispatch_metadata = SimpleNamespace(
            kv_indptr="owned_indptr",
            kv_indices="owned_indices",
            seq_lens="owned_seq_lens",
        )

        class Plan:
            metadata = SimpleNamespace(
                kv_indptr="plan_indptr",
                kv_indices="plan_indices",
            )
            paged_metadata = None
            kv_pool = FallbackPool()
            layer_idx = 0
            use_decode_attention = True

            def attention_dispatch_context(self, **kwargs):
                events.append(("context", kwargs))
                return TokenPoolAttentionDispatchContext(
                    layer_idx=0,
                    flat_metadata=dispatch_metadata,
                    paged_metadata=None,
                    token_kv_pool=self.kv_pool,
                    kv_buffer_owner=dispatch_owner,
                    workspace_owner=dispatch_owner,
                )

        plan = Plan()

        try:
            actual = _token_pool_backend_decode(
                native_forward,
                SimpleNamespace(num_key_value_groups=4, scaling=1.0),
                FakeQuery(),
                decode_metadata=plan.metadata,
                token_kv_pool=plan.kv_pool,
                layer_idx=0,
                token_pool_plan=plan,
            )
            stats = native_forward.token_pool_triton_stats()
        finally:
            if old_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = old_module
            if old_enable is None:
                os.environ.pop("WKVM_ENABLE_TOKEN_POOL_TRITON", None)
            else:
                os.environ["WKVM_ENABLE_TOKEN_POOL_TRITON"] = old_enable
            if old_disable is None:
                os.environ.pop("WKVM_DISABLE_TOKEN_POOL_TRITON", None)
            else:
                os.environ["WKVM_DISABLE_TOKEN_POOL_TRITON"] = old_disable
            native_forward.reset_token_pool_triton_stats(clear_disabled_shapes=True)

        self.assertEqual(actual, ("owned_triton", None))
        self.assertEqual(events[0][0], "context")
        self.assertEqual(events[1], "kv_buffers")
        self.assertEqual(events[2][0], "output")
        self.assertEqual(events[3], "decode")
        self.assertEqual(calls["key_buffer"], "owned_key")
        self.assertEqual(calls["value_buffer"], "owned_value")
        self.assertEqual(calls["kv_indptr"], "owned_indptr")
        self.assertEqual(calls["kv_indices"], "owned_indices")
        self.assertEqual(calls["output"], "owned_output")
        self.assertEqual(stats["attempts"], 1)
        self.assertEqual(stats["successes"], 1)

    def test_token_pool_reference_fallback_uses_plan_owned_context(self) -> None:
        import os
        from types import SimpleNamespace
        import wkvm.runner.gemma_native_forward as native_forward
        from wkvm.runner.gemma_token_pool import TokenPoolAttentionDispatchContext

        old_disable = os.environ.get("WKVM_DISABLE_TOKEN_POOL_TRITON")
        old_reference = native_forward._attention_forward_token_pool_gqa_reference
        native_forward.reset_token_pool_triton_stats(clear_disabled_shapes=True)
        events = []

        class FakeQuery:
            shape = (1, 8, 1, 512)
            dtype = "bfloat16"
            device = "cpu"
            is_cuda = False

        class ContextPool:
            pass

        context_pool = ContextPool()
        stale_pool = object()
        context_metadata = SimpleNamespace(
            kv_indptr="context_indptr",
            kv_indices="context_indices",
            out_cache_loc="context_write",
        )

        class WriteOwner:
            def store_current_kv(self, key_states, value_states):
                events.append(("store", key_states, value_states))
                return "context_write"

        class Plan:
            metadata = SimpleNamespace(
                kv_indptr="plan_indptr",
                kv_indices="plan_indices",
            )
            paged_metadata = None
            kv_pool = stale_pool
            layer_idx = 3
            use_decode_attention = True

            def attention_dispatch_context(self, **kwargs):
                events.append(("context", kwargs))
                return TokenPoolAttentionDispatchContext(
                    layer_idx=42,
                    flat_metadata=context_metadata,
                    paged_metadata=None,
                    token_kv_pool=context_pool,
                    kv_write_owner=WriteOwner(),
                )

        def reference(*_args, **kwargs):
            events.append(("reference", kwargs))
            return "reference", None

        try:
            os.environ["WKVM_DISABLE_TOKEN_POOL_TRITON"] = "1"
            native_forward._attention_forward_token_pool_gqa_reference = reference
            actual = _token_pool_backend_decode(
                native_forward,
                SimpleNamespace(num_key_value_groups=4, scaling=1.0),
                FakeQuery(),
                decode_metadata=SimpleNamespace(
                    kv_indptr="stale_indptr",
                    kv_indices="stale_indices",
                ),
                token_kv_pool=stale_pool,
                layer_idx=3,
                token_pool_plan=Plan(),
                current_key_states="current_key",
                current_value_states="current_value",
            )
        finally:
            native_forward._attention_forward_token_pool_gqa_reference = old_reference
            if old_disable is None:
                os.environ.pop("WKVM_DISABLE_TOKEN_POOL_TRITON", None)
            else:
                os.environ["WKVM_DISABLE_TOKEN_POOL_TRITON"] = old_disable
            native_forward.reset_token_pool_triton_stats(clear_disabled_shapes=True)

        self.assertEqual(actual, ("reference", None))
        self.assertEqual(events[0][0], "context")
        self.assertEqual(events[1], ("store", "current_key", "current_value"))
        self.assertEqual(events[2][0], "reference")
        self.assertIs(events[2][1]["decode_metadata"], context_metadata)
        self.assertIs(events[2][1]["token_kv_pool"], context_pool)
        self.assertEqual(events[2][1]["layer_idx"], 42)

    def test_token_pool_triton_auto_default_and_explicit_disable(self) -> None:
        import os
        import sys
        from types import ModuleType, SimpleNamespace
        import wkvm.runner.gemma_native_forward as native_forward

        native_forward.reset_token_pool_triton_stats(clear_disabled_shapes=True)
        module_name = "wkvm.runner.gemma_token_pool_triton"
        old_module = sys.modules.get(module_name)
        old_enable = os.environ.get("WKVM_ENABLE_TOKEN_POOL_TRITON")
        old_disable = os.environ.get("WKVM_DISABLE_TOKEN_POOL_TRITON")
        old_reference = native_forward._attention_forward_token_pool_gqa_reference
        fake_module = ModuleType(module_name)
        calls = {"triton": 0, "reference": 0}

        def decode(*args, **kwargs):
            calls["triton"] += 1
            return "triton"

        def reference(*args, **kwargs):
            calls["reference"] += 1
            return "reference", None

        fake_module.token_pool_gqa_decode = decode
        sys.modules[module_name] = fake_module
        os.environ.pop("WKVM_ENABLE_TOKEN_POOL_TRITON", None)
        os.environ.pop("WKVM_DISABLE_TOKEN_POOL_TRITON", None)
        native_forward._attention_forward_token_pool_gqa_reference = reference

        class FakeQuery:
            shape = (1, 8, 1, 512)
            dtype = "bfloat16"
            device = "cuda:0"
            is_cuda = True

        class FakePool:
            def get_kv_buffer(self, layer_idx):
                return "key_buffer", "value_buffer"

        try:
            auto = _token_pool_backend_decode(
                native_forward,
                SimpleNamespace(num_key_value_groups=4, scaling=1.0),
                FakeQuery(),
                decode_metadata=SimpleNamespace(kv_indptr="indptr", kv_indices="indices"),
                token_kv_pool=FakePool(),
                layer_idx=0,
            )
            os.environ["WKVM_DISABLE_TOKEN_POOL_TRITON"] = "1"
            disabled = _token_pool_backend_decode(
                native_forward,
                SimpleNamespace(num_key_value_groups=4, scaling=1.0),
                FakeQuery(),
                decode_metadata=SimpleNamespace(kv_indptr="indptr", kv_indices="indices"),
                token_kv_pool=FakePool(),
                layer_idx=0,
            )
            stats = native_forward.token_pool_triton_stats()
        finally:
            native_forward._attention_forward_token_pool_gqa_reference = old_reference
            if old_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = old_module
            if old_enable is None:
                os.environ.pop("WKVM_ENABLE_TOKEN_POOL_TRITON", None)
            else:
                os.environ["WKVM_ENABLE_TOKEN_POOL_TRITON"] = old_enable
            if old_disable is None:
                os.environ.pop("WKVM_DISABLE_TOKEN_POOL_TRITON", None)
            else:
                os.environ["WKVM_DISABLE_TOKEN_POOL_TRITON"] = old_disable
            native_forward.reset_token_pool_triton_stats(clear_disabled_shapes=True)

        self.assertEqual(auto, ("triton", None))
        self.assertEqual(disabled, ("reference", None))
        self.assertEqual(calls, {"triton": 1, "reference": 1})
        self.assertEqual(stats["calls"], 2)
        self.assertEqual(stats["attempts"], 1)
        self.assertEqual(stats["successes"], 1)
        self.assertEqual(stats["auto_enabled_calls"], 1)
        self.assertEqual(stats["effective_enabled_calls"], 1)
        self.assertEqual(stats["effective_disabled_calls"], 1)
        self.assertEqual(stats["env_enabled_calls"], 0)
        self.assertEqual(stats["env_disabled_calls"], 1)
        self.assertTrue(stats["env_disabled"])
        self.assertFalse(stats["effective_enabled"])

    def test_token_pool_triton_dispatch_plan_tracks_env_changes(self) -> None:
        import os
        import wkvm.runner.gemma_native_forward as native_forward

        names = (
            "WKVM_ENABLE_TOKEN_POOL_TRITON",
            "WKVM_DISABLE_TOKEN_POOL_TRITON",
            "WKVM_ENABLE_TOKEN_POOL_PAGED_TRITON",
            "WKVM_ENABLE_TOKEN_POOL_SPLIT_TRITON",
            "WKVM_TOKEN_POOL_TRITON_SPLIT_KV",
            "WKVM_ENABLE_TOKEN_POOL_PAGED_SPLIT_TRITON",
            "WKVM_TOKEN_POOL_TRITON_PAGED_SPLIT_KV",
            "WKVM_TOKEN_POOL_TRITON_INPUT_PRECISION",
            "WKVM_TOKEN_POOL_TRITON_DOT_DTYPE",
            "WKVM_TOKEN_POOL_TRITON_STRICT",
        )
        old_env = {name: os.environ.get(name) for name in names}
        try:
            for name in names:
                os.environ.pop(name, None)
            native_forward.reset_token_pool_triton_stats(clear_disabled_shapes=True)
            auto = native_forward._token_pool_triton_dispatch_plan()
            self.assertTrue(auto.effective_enabled)
            self.assertTrue(auto.auto_default_enabled)
            self.assertFalse(auto.paged_enabled)

            os.environ["WKVM_DISABLE_TOKEN_POOL_TRITON"] = "1"
            disabled = native_forward._token_pool_triton_dispatch_plan()
            self.assertFalse(disabled.effective_enabled)
            self.assertFalse(disabled.auto_default_enabled)
            self.assertTrue(disabled.env_disabled)

            os.environ.pop("WKVM_DISABLE_TOKEN_POOL_TRITON", None)
            os.environ["WKVM_ENABLE_TOKEN_POOL_TRITON"] = "1"
            os.environ["WKVM_ENABLE_TOKEN_POOL_SPLIT_TRITON"] = "1"
            os.environ["WKVM_ENABLE_TOKEN_POOL_PAGED_TRITON"] = "1"
            os.environ["WKVM_ENABLE_TOKEN_POOL_PAGED_SPLIT_TRITON"] = "1"
            os.environ["WKVM_TOKEN_POOL_TRITON_INPUT_PRECISION"] = "ieee"
            os.environ["WKVM_TOKEN_POOL_TRITON_DOT_DTYPE"] = "native"
            os.environ["WKVM_TOKEN_POOL_TRITON_STRICT"] = "yes"
            explicit = native_forward._token_pool_triton_dispatch_plan()
            self.assertTrue(explicit.env_enabled)
            self.assertTrue(explicit.effective_enabled)
            self.assertFalse(explicit.auto_default_enabled)
            self.assertTrue(explicit.paged_enabled)
            self.assertTrue(explicit.split_enabled)
            self.assertTrue(explicit.paged_split_enabled)
            self.assertEqual(explicit.input_precision_policy, "ieee")
            self.assertEqual(explicit.dot_dtype_policy, "native")
            self.assertTrue(explicit.strict)
        finally:
            for name, value in old_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            native_forward.reset_token_pool_triton_stats(clear_disabled_shapes=True)

    def test_token_pool_paged_triton_dispatch_is_explicit(self) -> None:
        import os
        import sys
        from types import ModuleType, SimpleNamespace
        import wkvm.runner.gemma_native_forward as native_forward

        native_forward.reset_token_pool_triton_stats(clear_disabled_shapes=True)
        module_name = "wkvm.runner.gemma_token_pool_triton"
        old_module = sys.modules.get(module_name)
        old_enable = os.environ.get("WKVM_ENABLE_TOKEN_POOL_TRITON")
        old_paged = os.environ.get("WKVM_ENABLE_TOKEN_POOL_PAGED_TRITON")
        old_disable = os.environ.get("WKVM_DISABLE_TOKEN_POOL_TRITON")
        fake_module = ModuleType(module_name)
        calls = {"flat": 0, "paged": 0, "paged_block_size": None}

        def flat_decode(*args, **kwargs):
            calls["flat"] += 1
            return "flat"

        def paged_decode(*args, **kwargs):
            calls["paged"] += 1
            calls["paged_block_size"] = kwargs.get("block_size")
            return "paged"

        fake_module.token_pool_gqa_decode = flat_decode
        fake_module.token_pool_paged_gqa_decode = paged_decode
        sys.modules[module_name] = fake_module
        os.environ["WKVM_ENABLE_TOKEN_POOL_TRITON"] = "1"
        os.environ["WKVM_ENABLE_TOKEN_POOL_PAGED_TRITON"] = "1"
        os.environ.pop("WKVM_DISABLE_TOKEN_POOL_TRITON", None)

        class FakeQuery:
            shape = (1, 8, 1, 512)
            dtype = "bfloat16"
            device = "cuda:0"
            is_cuda = True

        class FakePool:
            def get_kv_buffer(self, layer_idx):
                return "key_buffer", "value_buffer"

        metadata = SimpleNamespace(
            req_pool_indices="reqs",
            seq_lens="seq_lens",
            logical_seq_lens="logical_seq_lens",
            out_cache_loc="out_cache_loc",
            block_tables="block_tables",
            block_table_lens="block_table_lens",
            selected_start_positions="selected_start_positions",
            block_size=32,
        )
        try:
            actual = _token_pool_backend_decode(
                native_forward,
                SimpleNamespace(num_key_value_groups=4, scaling=1.0),
                FakeQuery(),
                decode_metadata=metadata,
                token_kv_pool=FakePool(),
                layer_idx=0,
            )
            stats = native_forward.token_pool_triton_stats()
        finally:
            if old_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = old_module
            if old_enable is None:
                os.environ.pop("WKVM_ENABLE_TOKEN_POOL_TRITON", None)
            else:
                os.environ["WKVM_ENABLE_TOKEN_POOL_TRITON"] = old_enable
            if old_paged is None:
                os.environ.pop("WKVM_ENABLE_TOKEN_POOL_PAGED_TRITON", None)
            else:
                os.environ["WKVM_ENABLE_TOKEN_POOL_PAGED_TRITON"] = old_paged
            if old_disable is None:
                os.environ.pop("WKVM_DISABLE_TOKEN_POOL_TRITON", None)
            else:
                os.environ["WKVM_DISABLE_TOKEN_POOL_TRITON"] = old_disable
            native_forward.reset_token_pool_triton_stats(clear_disabled_shapes=True)

        self.assertEqual(actual, ("paged", None))
        self.assertEqual(calls, {"flat": 0, "paged": 1, "paged_block_size": 32})
        self.assertEqual(stats["calls"], 1)
        self.assertEqual(stats["attempts"], 1)
        self.assertEqual(stats["successes"], 1)
        self.assertEqual(stats["paged_enabled_calls"], 1)
        self.assertEqual(stats["paged_attempts"], 1)
        self.assertEqual(stats["paged_successes"], 1)

    def test_token_pool_triton_attention_matches_ragged_manual_gqa_on_cuda(self) -> None:
        import os
        import torch
        from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer
        from wkvm.runner.gemma_native_forward import (
            NativeGemma4TextDecoderLayer,
            _attention_forward,
            _attention_forward_manual_gqa,
        )
        from wkvm.runner.gemma_token_pool import (
            DecodeBatchMetadata,
            TokenKVLayerSpec,
            TokenKVPool,
        )

        if not torch.cuda.is_available():
            self.skipTest("CUDA is required for Triton token-pool attention")
        from wkvm.runner.gemma_token_pool_triton import token_pool_gqa_decode

        torch.manual_seed(20)
        cfg = Gemma4TextConfig(
            vocab_size=64,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=16,
            hidden_size_per_layer_input=4,
            vocab_size_per_layer_input=64,
            sliding_window=16,
            layer_types=["sliding_attention", "sliding_attention"],
            num_kv_shared_layers=0,
            attention_dropout=0.0,
            attention_bias=False,
            global_head_dim=16,
        )
        hf_layer = Gemma4TextDecoderLayer(cfg, layer_idx=0).eval().cuda()
        native_layer = NativeGemma4TextDecoderLayer(
            hf_layer,
            native_attention_backend="manual_gqa",
        )
        pool = TokenKVPool(
            capacity=12,
            layer_specs=[
                TokenKVLayerSpec(
                    layer_id=0,
                    num_kv_heads=cfg.num_key_value_heads,
                    head_dim=cfg.head_dim,
                    dtype=torch.float32,
                )
            ],
            dtype=torch.float32,
            device="cuda",
        )
        slots = pool.alloc_slots(12)
        keys = torch.randn(12, cfg.num_key_value_heads, cfg.head_dim, device="cuda")
        values = torch.randn(12, cfg.num_key_value_heads, cfg.head_dim, device="cuda")
        pool.set_kv(0, slots, keys, values)
        query = torch.randn(
            3,
            cfg.num_attention_heads,
            1,
            cfg.head_dim,
            device="cuda",
        )
        metadata = DecodeBatchMetadata(
            req_pool_indices=torch.tensor([0, 1, 2], dtype=torch.int32, device="cuda"),
            seq_lens=torch.tensor([4, 3, 5], dtype=torch.int32, device="cuda"),
            logical_seq_lens=torch.tensor([4, 3, 5], dtype=torch.int32, device="cuda"),
            out_cache_loc=torch.tensor([3, 6, 11], dtype=torch.int32, device="cuda"),
            kv_indptr=torch.tensor([0, 4, 7, 12], dtype=torch.int32, device="cuda"),
            kv_indices=torch.arange(12, dtype=torch.int32, device="cuda"),
        )

        with torch.inference_mode():
            direct = token_pool_gqa_decode(
                query,
                pool.get_kv_buffer(0)[0],
                pool.get_kv_buffer(0)[1],
                metadata.kv_indptr,
                metadata.kv_indices,
                num_key_value_groups=native_layer.attn_meta.num_key_value_groups,
                scaling=native_layer.attn_meta.scaling,
            )
            old_flag = os.environ.get("WKVM_ENABLE_TOKEN_POOL_TRITON")
            os.environ["WKVM_ENABLE_TOKEN_POOL_TRITON"] = "1"
            try:
                actual, actual_weights = _attention_forward(
                    native_layer.attn_meta,
                    query,
                    keys[:1].permute(1, 0, 2).unsqueeze(0),
                    values[:1].permute(1, 0, 2).unsqueeze(0),
                    None,
                    backend="manual_gqa",
                    decode_metadata=metadata,
                    token_kv_pool=pool,
                    layer_idx=0,
                )
            finally:
                if old_flag is None:
                    os.environ.pop("WKVM_ENABLE_TOKEN_POOL_TRITON", None)
                else:
                    os.environ["WKVM_ENABLE_TOKEN_POOL_TRITON"] = old_flag
            expected_rows = []
            for start, end, row in ((0, 4, 0), (4, 7, 1), (7, 12, 2)):
                expected, _ = _attention_forward_manual_gqa(
                    native_layer.attn_meta,
                    query[row : row + 1],
                    keys[start:end].permute(1, 0, 2).unsqueeze(0),
                    values[start:end].permute(1, 0, 2).unsqueeze(0),
                    None,
                )
                expected_rows.append(expected)
            expected = torch.cat(expected_rows, dim=0)
        torch.cuda.synchronize()

        self.assertIsNone(actual_weights)
        self.assertLess((expected - direct).abs().max().item(), 1e-5)
        self.assertLess((expected - actual).abs().max().item(), 1e-5)

    def test_decode_layer_token_pool_context_matches_dense_cache(self) -> None:
        import torch
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextDecoderLayer,
            Gemma4TextRotaryEmbedding,
        )
        from wkvm.models.gemma import GemmaRoutedSpanConfig
        from wkvm.runner.gemma_native_forward import NativeGemma4TextDecoderLayer
        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache
        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenPoolAttentionBinding,
            TokenKVLayerSpec,
            TokenKVPool,
            build_decode_metadata_from_token_slot_rows,
        )

        torch.manual_seed(21)
        cfg = _tiny_config()
        hf_layer = Gemma4TextDecoderLayer(cfg, layer_idx=0).eval()
        dense_layer = NativeGemma4TextDecoderLayer(
            hf_layer,
            native_attention_backend="manual_gqa",
        )
        token_pool_layer = NativeGemma4TextDecoderLayer(
            hf_layer,
            native_attention_backend="manual_gqa",
        )
        rotary = Gemma4TextRotaryEmbedding(cfg)
        native_cfg = GemmaRoutedSpanConfig(
            num_hidden_layers=cfg.num_hidden_layers,
            num_kv_shared_layers=cfg.num_kv_shared_layers,
            layer_types=tuple(cfg.layer_types),
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            sliding_window=cfg.sliding_window,
        )
        dense_cache = NativeGemmaRoutedCache(cfg, native_cfg)
        token_pool_cache = NativeGemmaRoutedCache(cfg, native_cfg)

        prefill_hidden = torch.randn(1, 3, cfg.hidden_size)
        prefill_ple = torch.randn(1, 3, cfg.hidden_size_per_layer_input)
        prefill_pos = torch.arange(3).unsqueeze(0)
        prefill_pos_emb = rotary(prefill_hidden, prefill_pos, "sliding_attention")
        prefill_mask = _causal_mask(1, 3, dtype=prefill_hidden.dtype, device=prefill_hidden.device)

        decode_hidden = torch.randn(1, 1, cfg.hidden_size)
        decode_ple = torch.randn(1, 1, cfg.hidden_size_per_layer_input)
        decode_pos = torch.tensor([[3]])
        decode_pos_emb = rotary(decode_hidden, decode_pos, "sliding_attention")

        pool = TokenKVPool(
            capacity=4,
            layer_specs=[
                TokenKVLayerSpec(
                    layer_id=0,
                    num_kv_heads=cfg.num_key_value_heads,
                    head_dim=cfg.head_dim,
                    dtype=torch.float32,
                )
            ],
            dtype=torch.float32,
        )
        table = ReqToTokenTable(max_requests=1, max_context_len=4)
        req_slot = table.allocate("r0")
        token_slots = pool.alloc_slots(4)
        prefill_slots = token_slots[:3]
        decode_slot = token_slots[3:4]
        table.append_slots(req_slot, prefill_slots)

        with torch.inference_mode():
            dense_layer(
                prefill_hidden,
                prefill_ple,
                shared_kv_states=UserDict(),
                position_embeddings=prefill_pos_emb,
                attention_mask=prefill_mask,
                position_ids=prefill_pos,
                past_key_values=dense_cache,
            )
            token_pool_layer(
                prefill_hidden,
                prefill_ple,
                shared_kv_states=UserDict(),
                position_embeddings=prefill_pos_emb,
                attention_mask=prefill_mask,
                position_ids=prefill_pos,
                past_key_values=token_pool_cache,
            )

        prefill_keys = token_pool_cache.layers[0].keys[0].permute(1, 0, 2).contiguous()
        prefill_values = token_pool_cache.layers[0].values[0].permute(1, 0, 2).contiguous()
        pool.set_kv(0, prefill_slots, prefill_keys, prefill_values)
        table.append_slots(req_slot, decode_slot)
        metadata = table.build_decode_metadata(
            [req_slot],
            out_cache_loc=decode_slot,
        )
        wrong_type_metadata = build_decode_metadata_from_token_slot_rows(
            [[int(decode_slot.reshape(-1)[0].item())]],
            out_cache_loc=decode_slot,
        )

        class BindingOnlyContext:
            kv_pool = pool
            metadata_by_layer_type = {"sliding_attention": wrong_type_metadata}

            def attention_binding_for_layer(
                self,
                layer_idx,
                layer_type,
                *,
                attention_mask_present: bool = False,
            ):
                self.last_request = (
                    layer_idx,
                    layer_type,
                    attention_mask_present,
                )
                return TokenPoolAttentionBinding(
                    layer_idx=layer_idx,
                    metadata=metadata,
                    paged_metadata=None,
                    kv_pool=pool,
                )

            def metadata_for_layer(self, *_args, **_kwargs):
                raise AssertionError("legacy metadata lookup should not run")

        context = BindingOnlyContext()
        kv_set_calls_before_decode = pool.kv_set_calls

        with torch.inference_mode():
            expected = dense_layer(
                decode_hidden,
                decode_ple,
                shared_kv_states=UserDict(),
                position_embeddings=decode_pos_emb,
                attention_mask=None,
                position_ids=decode_pos,
                past_key_values=dense_cache,
            )
            actual = token_pool_layer(
                decode_hidden,
                decode_ple,
                shared_kv_states=UserDict(),
                position_embeddings=decode_pos_emb,
                attention_mask=None,
                position_ids=decode_pos,
                past_key_values=token_pool_cache,
                wkvm_token_pool_decode=context,
            )

        pooled_key, pooled_value = pool.gather_kv(0, decode_slot)
        dense_decode_key = dense_cache.layers[0].keys[0, :, -1:, :].permute(1, 0, 2)
        dense_decode_value = dense_cache.layers[0].values[0, :, -1:, :].permute(1, 0, 2)
        self.assertEqual(context.last_request, (0, "sliding_attention", False))
        self.assertEqual(pool.kv_set_calls, kv_set_calls_before_decode + 1)
        self.assertLess((expected - actual).abs().max().item(), 1e-6)
        self.assertLess((pooled_key - dense_decode_key).abs().max().item(), 1e-6)
        self.assertLess((pooled_value - dense_decode_value).abs().max().item(), 1e-6)

    def test_token_pool_decode_skips_dense_cache_update_when_covered(self) -> None:
        import torch
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextDecoderLayer,
            Gemma4TextRotaryEmbedding,
        )
        from wkvm.models.gemma import GemmaRoutedSpanConfig
        from wkvm.runner.gemma_native_forward import NativeGemma4TextDecoderLayer
        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache
        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenKVLayerSpec,
            TokenKVPool,
            TokenPoolDecodeContext,
        )

        class RaisingDecodeCache:
            def update(self, *_args, **_kwargs):
                raise AssertionError("dense cache update should be skipped")

        torch.manual_seed(22)
        cfg = _tiny_config()
        hf_layer = Gemma4TextDecoderLayer(cfg, layer_idx=0).eval()
        dense_layer = NativeGemma4TextDecoderLayer(
            hf_layer,
            native_attention_backend="manual_gqa",
        )
        token_pool_layer = NativeGemma4TextDecoderLayer(
            hf_layer,
            native_attention_backend="manual_gqa",
        )
        rotary = Gemma4TextRotaryEmbedding(cfg)
        native_cfg = GemmaRoutedSpanConfig(
            num_hidden_layers=cfg.num_hidden_layers,
            num_kv_shared_layers=cfg.num_kv_shared_layers,
            layer_types=tuple(cfg.layer_types),
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            sliding_window=cfg.sliding_window,
        )
        dense_cache = NativeGemmaRoutedCache(cfg, native_cfg)
        prefill_cache = NativeGemmaRoutedCache(cfg, native_cfg)

        prefill_hidden = torch.randn(1, 3, cfg.hidden_size)
        prefill_ple = torch.randn(1, 3, cfg.hidden_size_per_layer_input)
        prefill_pos = torch.arange(3).unsqueeze(0)
        prefill_pos_emb = rotary(prefill_hidden, prefill_pos, "sliding_attention")
        prefill_mask = _causal_mask(1, 3, dtype=prefill_hidden.dtype, device=prefill_hidden.device)

        decode_hidden = torch.randn(1, 1, cfg.hidden_size)
        decode_ple = torch.randn(1, 1, cfg.hidden_size_per_layer_input)
        decode_pos = torch.tensor([[3]])
        decode_pos_emb = rotary(decode_hidden, decode_pos, "sliding_attention")

        pool = TokenKVPool(
            capacity=4,
            layer_specs=[
                TokenKVLayerSpec(
                    layer_id=0,
                    num_kv_heads=cfg.num_key_value_heads,
                    head_dim=cfg.head_dim,
                    dtype=torch.float32,
                )
            ],
            dtype=torch.float32,
        )
        table = ReqToTokenTable(max_requests=1, max_context_len=4)
        req_slot = table.allocate("r0")
        token_slots = pool.alloc_slots(4)
        prefill_slots = token_slots[:3]
        decode_slot = token_slots[3:4]
        table.append_slots(req_slot, prefill_slots)

        with torch.inference_mode():
            dense_layer(
                prefill_hidden,
                prefill_ple,
                shared_kv_states=UserDict(),
                position_embeddings=prefill_pos_emb,
                attention_mask=prefill_mask,
                position_ids=prefill_pos,
                past_key_values=dense_cache,
            )
            token_pool_layer(
                prefill_hidden,
                prefill_ple,
                shared_kv_states=UserDict(),
                position_embeddings=prefill_pos_emb,
                attention_mask=prefill_mask,
                position_ids=prefill_pos,
                past_key_values=prefill_cache,
            )

        prefill_keys = prefill_cache.layers[0].keys[0].permute(1, 0, 2).contiguous()
        prefill_values = prefill_cache.layers[0].values[0].permute(1, 0, 2).contiguous()
        pool.set_kv(0, prefill_slots, prefill_keys, prefill_values)
        table.append_slots(req_slot, decode_slot)
        metadata = table.build_decode_metadata(
            [req_slot],
            out_cache_loc=decode_slot,
        )
        context = TokenPoolDecodeContext(
            metadata_by_layer_type={"sliding_attention": metadata},
            kv_pool=pool,
        )

        with torch.inference_mode():
            expected = dense_layer(
                decode_hidden,
                decode_ple,
                shared_kv_states=UserDict(),
                position_embeddings=decode_pos_emb,
                attention_mask=None,
                position_ids=decode_pos,
                past_key_values=dense_cache,
            )
            actual = token_pool_layer(
                decode_hidden,
                decode_ple,
                shared_kv_states=UserDict(),
                position_embeddings=decode_pos_emb,
                attention_mask=None,
                position_ids=decode_pos,
                past_key_values=RaisingDecodeCache(),
                wkvm_token_pool_decode=context,
            )

        self.assertLess((expected - actual).abs().max().item(), 1e-6)

    def test_shared_kv_token_pool_binding_does_not_store_current_kv(self) -> None:
        import torch
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextDecoderLayer,
            Gemma4TextRotaryEmbedding,
        )
        from wkvm.runner.gemma_native_forward import NativeGemma4TextDecoderLayer
        from wkvm.runner.gemma_token_pool import (
            TokenKVLayerSpec,
            TokenKVPool,
            build_decode_metadata_from_token_slot_rows,
        )

        torch.manual_seed(24)
        cfg = _tiny_shared_config()
        hf_layer = Gemma4TextDecoderLayer(cfg, layer_idx=2).eval()
        native_layer = NativeGemma4TextDecoderLayer(
            hf_layer,
            native_attention_backend="manual_gqa",
        )
        rotary = Gemma4TextRotaryEmbedding(cfg)
        hidden = torch.randn(1, 1, cfg.hidden_size)
        per_layer_input = torch.randn(1, 1, cfg.hidden_size_per_layer_input)
        position_ids = torch.tensor([[3]])
        position_embeddings = rotary(hidden, position_ids, "sliding_attention")
        shared_keys = torch.randn(1, cfg.num_key_value_heads, 3, cfg.head_dim)
        shared_values = torch.randn(1, cfg.num_key_value_heads, 3, cfg.head_dim)

        pool = TokenKVPool(
            capacity=3,
            layer_specs=[
                TokenKVLayerSpec(
                    layer_id=0,
                    num_kv_heads=cfg.num_key_value_heads,
                    head_dim=cfg.head_dim,
                    dtype=torch.float32,
                ),
                TokenKVLayerSpec(
                    layer_id=2,
                    num_kv_heads=cfg.num_key_value_heads,
                    head_dim=cfg.head_dim,
                    dtype=torch.float32,
                    kv_share_target_layer=0,
                )
            ],
            dtype=torch.float32,
        )
        slots = pool.alloc_slots(3)
        pool.set_kv(
            0,
            slots,
            shared_keys[0].permute(1, 0, 2).contiguous(),
            shared_values[0].permute(1, 0, 2).contiguous(),
        )
        metadata = build_decode_metadata_from_token_slot_rows(
            [slots],
            out_cache_loc=slots[-1:],
        )

        class NoStoreBinding:
            def __init__(self) -> None:
                self.metadata = metadata
                self.paged_metadata = None
                self.kv_pool = pool
                self.store_calls = 0

            def store_current_kv(self, *_args, **_kwargs):
                self.store_calls += 1
                raise AssertionError("shared-KV layer should not store current KV")

        binding = NoStoreBinding()

        class BindingContext:
            def attention_binding_for_layer(
                self,
                layer_idx,
                layer_type,
                *,
                attention_mask_present: bool = False,
            ):
                self.last_request = (
                    layer_idx,
                    layer_type,
                    attention_mask_present,
                )
                return binding

        context = BindingContext()

        with torch.inference_mode():
            actual = native_layer(
                hidden,
                per_layer_input,
                shared_kv_states=UserDict(
                    {"sliding_attention": (shared_keys, shared_values)}
                ),
                position_embeddings=position_embeddings,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=None,
                wkvm_token_pool_decode=context,
            )

        self.assertEqual(tuple(actual.shape), tuple(hidden.shape))
        self.assertEqual(pool.target_layer(2), 0)
        self.assertEqual(context.last_request, (0, "sliding_attention", False))
        self.assertEqual(binding.store_calls, 0)

    def test_shared_kv_token_pool_decode_uses_source_layer_alias(self) -> None:
        import torch
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextDecoderLayer,
            Gemma4TextRotaryEmbedding,
        )
        from wkvm.runner.gemma_native_forward import NativeGemma4TextDecoderLayer
        from wkvm.runner.gemma_token_pool import (
            TokenKVLayerSpec,
            TokenKVPool,
            build_decode_metadata_from_token_slot_rows,
        )

        torch.manual_seed(25)
        cfg = _tiny_shared_config()
        hf_layer = Gemma4TextDecoderLayer(cfg, layer_idx=2).eval()
        native_layer = NativeGemma4TextDecoderLayer(
            hf_layer,
            native_attention_backend="manual_gqa",
        )
        rotary = Gemma4TextRotaryEmbedding(cfg)
        hidden = torch.randn(1, 1, cfg.hidden_size)
        per_layer_input = torch.randn(1, 1, cfg.hidden_size_per_layer_input)
        position_ids = torch.tensor([[3]])
        position_embeddings = rotary(hidden, position_ids, "sliding_attention")
        shared_keys = torch.randn(1, cfg.num_key_value_heads, 3, cfg.head_dim)
        shared_values = torch.randn(1, cfg.num_key_value_heads, 3, cfg.head_dim)

        pool = TokenKVPool(
            capacity=3,
            layer_specs=[
                TokenKVLayerSpec(
                    layer_id=0,
                    num_kv_heads=cfg.num_key_value_heads,
                    head_dim=cfg.head_dim,
                    dtype=torch.float32,
                ),
                TokenKVLayerSpec(
                    layer_id=2,
                    num_kv_heads=cfg.num_key_value_heads,
                    head_dim=cfg.head_dim,
                    dtype=torch.float32,
                    kv_share_target_layer=0,
                ),
            ],
            dtype=torch.float32,
        )
        slots = pool.alloc_slots(3)
        pool.set_kv(
            0,
            slots,
            shared_keys[0].permute(1, 0, 2).contiguous(),
            shared_values[0].permute(1, 0, 2).contiguous(),
        )
        metadata = build_decode_metadata_from_token_slot_rows(
            [slots],
            out_cache_loc=slots[-1:],
        )

        class NoStoreBinding:
            def __init__(self) -> None:
                self.metadata = metadata
                self.paged_metadata = None
                self.kv_pool = pool
                self.store_calls = 0

            def store_current_kv(self, *_args, **_kwargs):
                self.store_calls += 1
                raise AssertionError("shared-KV alias layer should not write KV")

        binding = NoStoreBinding()

        class BindingContext:
            def attention_binding_for_layer(
                self,
                layer_idx,
                layer_type,
                *,
                attention_mask_present: bool = False,
            ):
                self.last_request = (
                    layer_idx,
                    layer_type,
                    attention_mask_present,
                )
                return binding

        context = BindingContext()

        with torch.inference_mode():
            expected = native_layer(
                hidden,
                per_layer_input,
                shared_kv_states=UserDict(
                    {"sliding_attention": (shared_keys, shared_values)}
                ),
                position_embeddings=position_embeddings,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=None,
            )
            actual = native_layer(
                hidden,
                per_layer_input,
                shared_kv_states=UserDict(),
                position_embeddings=position_embeddings,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=None,
                wkvm_token_pool_decode=context,
            )

        self.assertEqual(pool.target_layer(2), 0)
        self.assertEqual(context.last_request, (0, "sliding_attention", False))
        self.assertEqual(binding.store_calls, 0)
        self.assertLess((expected - actual).abs().max().item(), 1e-6)

    def test_full_attention_token_pool_context_matches_routed_dense_cache(self) -> None:
        import torch
        from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextDecoderLayer,
            Gemma4TextRotaryEmbedding,
        )
        from wkvm.models.gemma import GemmaRoutedSpanConfig
        from wkvm.runner.gemma_native_forward import NativeGemma4TextDecoderLayer
        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache
        from wkvm.runner.gemma_token_pool import (
            TokenKVLayerSpec,
            TokenKVPool,
            TokenPoolDecodeContext,
            build_decode_metadata_from_token_slot_rows,
        )

        torch.manual_seed(23)
        cfg = Gemma4TextConfig(
            vocab_size=64,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            hidden_size_per_layer_input=4,
            vocab_size_per_layer_input=64,
            sliding_window=8,
            layer_types=["full_attention"],
            num_kv_shared_layers=0,
            attention_dropout=0.0,
            attention_bias=False,
            global_head_dim=4,
        )
        hf_layer = Gemma4TextDecoderLayer(cfg, layer_idx=0).eval()
        dense_layer = NativeGemma4TextDecoderLayer(
            hf_layer,
            native_attention_backend="manual_gqa",
        )
        token_pool_layer = NativeGemma4TextDecoderLayer(
            hf_layer,
            native_attention_backend="manual_gqa",
        )
        rotary = Gemma4TextRotaryEmbedding(cfg)
        native_cfg = GemmaRoutedSpanConfig(
            num_hidden_layers=1,
            num_kv_shared_layers=0,
            layer_types=tuple(cfg.layer_types),
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.global_head_dim,
            sliding_window=cfg.sliding_window,
            sink_tokens=1,
            ring_tokens=1,
            routed_slots=2,
            reps_per_slot=1,
            span_budget_tokens=4,
            pending_tokens=2,
            max_span_tokens=3,
        )
        dense_cache = NativeGemmaRoutedCache(cfg, native_cfg)
        token_pool_cache = NativeGemmaRoutedCache(cfg, native_cfg)

        prefill_hidden = torch.randn(1, 5, cfg.hidden_size)
        prefill_ple = torch.randn(1, 5, cfg.hidden_size_per_layer_input)
        prefill_pos = torch.arange(5).unsqueeze(0)
        prefill_pos_emb = rotary(prefill_hidden, prefill_pos, "full_attention")
        prefill_mask = _causal_mask(1, 5, dtype=prefill_hidden.dtype, device=prefill_hidden.device)

        decode_hidden = torch.randn(1, 1, cfg.hidden_size)
        decode_ple = torch.randn(1, 1, cfg.hidden_size_per_layer_input)
        decode_pos = torch.tensor([[5]])
        decode_pos_emb = rotary(decode_hidden, decode_pos, "full_attention")

        with torch.inference_mode():
            dense_layer(
                prefill_hidden,
                prefill_ple,
                shared_kv_states=UserDict(),
                position_embeddings=prefill_pos_emb,
                attention_mask=prefill_mask,
                position_ids=prefill_pos,
                past_key_values=dense_cache,
            )
            token_pool_layer(
                prefill_hidden,
                prefill_ple,
                shared_kv_states=UserDict(),
                position_embeddings=prefill_pos_emb,
                attention_mask=prefill_mask,
                position_ids=prefill_pos,
                past_key_values=token_pool_cache,
            )

        materialized_width = token_pool_cache.layers[0].materialized_tokens()
        self.assertGreater(materialized_width, 5)
        pool = TokenKVPool(
            capacity=materialized_width + 1,
            layer_specs=[
                TokenKVLayerSpec(
                    layer_id=0,
                    num_kv_heads=cfg.num_key_value_heads,
                    head_dim=cfg.global_head_dim,
                    dtype=torch.float32,
                )
            ],
            dtype=torch.float32,
        )
        slots = pool.alloc_slots(materialized_width + 1)
        prefill_slots = slots[:materialized_width].flip(0)
        decode_slot = slots[materialized_width : materialized_width + 1]
        token_pool_cache.layers[0].write_materialized_readout_to_token_pool(
            pool,
            prefill_slots,
        )
        slot_row = torch.cat([prefill_slots, decode_slot])
        metadata = build_decode_metadata_from_token_slot_rows(
            [slot_row],
            logical_seq_lens=[dense_cache.layers[0].cumulative_length + 1],
            out_cache_loc=decode_slot,
        )
        context = TokenPoolDecodeContext(
            metadata_by_layer_type={"full_attention": metadata},
            metadata_by_layer_id={0: metadata},
            kv_pool=pool,
        )

        with torch.inference_mode():
            expected = dense_layer(
                decode_hidden,
                decode_ple,
                shared_kv_states=UserDict(),
                position_embeddings=decode_pos_emb,
                attention_mask=None,
                position_ids=decode_pos,
                past_key_values=dense_cache,
            )
            actual = token_pool_layer(
                decode_hidden,
                decode_ple,
                shared_kv_states=UserDict(),
                position_embeddings=decode_pos_emb,
                attention_mask=None,
                position_ids=decode_pos,
                past_key_values=token_pool_cache,
                wkvm_token_pool_decode=context,
            )

        pooled_key, pooled_value = pool.gather_kv(0, decode_slot)
        dense_decode_key = dense_cache.layers[0].keys[0, :, -1:, :].permute(1, 0, 2)
        dense_decode_value = dense_cache.layers[0].values[0, :, -1:, :].permute(1, 0, 2)
        self.assertLess((expected - actual).abs().max().item(), 1e-6)
        self.assertLess((pooled_key - dense_decode_key).abs().max().item(), 1e-6)
        self.assertLess((pooled_value - dense_decode_value).abs().max().item(), 1e-6)

    def test_text_prefix_with_wkvm_cache_matches_hf_layers(self) -> None:
        import torch
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel
        from wkvm.models.gemma import GemmaRoutedSpanConfig
        from wkvm.runner.gemma_native_forward import NativeGemma4TextPrefix
        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        torch.manual_seed(23)
        cfg = _tiny_config()
        text_model = Gemma4TextModel(cfg).eval()
        native_prefix = NativeGemma4TextPrefix(text_model, num_layers=2)
        native_cfg = GemmaRoutedSpanConfig(
            num_hidden_layers=cfg.num_hidden_layers,
            num_kv_shared_layers=cfg.num_kv_shared_layers,
            layer_types=tuple(cfg.layer_types),
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            sliding_window=cfg.sliding_window,
        )
        hf_cache = NativeGemmaRoutedCache(cfg, native_cfg)
        native_cache = NativeGemmaRoutedCache(cfg, native_cfg)

        input_ids = torch.tensor([[1, 7, 9, 11, 13]])
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
        mask = _causal_mask(1, input_ids.shape[1], dtype=torch.float32, device=input_ids.device)
        mask_mapping = {"sliding_attention": mask, "full_attention": mask}
        decode_ids = torch.tensor([[17]])
        decode_position_ids = torch.tensor([[input_ids.shape[1]]])

        def hf_prefix(ids, pos, cache, attn_mask):
            inputs_embeds = text_model.embed_tokens(ids)
            ple = text_model.get_per_layer_inputs(ids, inputs_embeds)
            ple = text_model.project_per_layer_inputs(inputs_embeds, ple)
            position_embeddings = {
                layer_type: text_model.rotary_emb(inputs_embeds, pos, layer_type)
                for layer_type in set(cfg.layer_types)
            }
            hidden = inputs_embeds
            shared_kv_states = UserDict()
            for i, layer in enumerate(text_model.layers[:2]):
                layer_type = cfg.layer_types[i]
                hidden = layer(
                    hidden,
                    ple[:, :, i, :],
                    shared_kv_states=shared_kv_states,
                    position_embeddings=position_embeddings[layer_type],
                    attention_mask=attn_mask[layer_type],
                    position_ids=pos,
                    past_key_values=cache,
                )
            return hidden

        with torch.inference_mode():
            hf_prefill = hf_prefix(input_ids, position_ids, hf_cache, mask_mapping)
            native_prefill = native_prefix(
                input_ids=input_ids,
                attention_mask=mask_mapping,
                position_ids=position_ids,
                past_key_values=native_cache,
            ).hidden_states
            hf_decode = hf_prefix(
                decode_ids,
                decode_position_ids,
                hf_cache,
                {"sliding_attention": None, "full_attention": None},
            )
            native_decode = native_prefix(
                input_ids=decode_ids,
                attention_mask={"sliding_attention": None, "full_attention": None},
                position_ids=decode_position_ids,
                past_key_values=native_cache,
            ).hidden_states

        self.assertLess((hf_prefill - native_prefill).abs().max().item(), 5e-6)
        self.assertLess((hf_decode - native_decode).abs().max().item(), 5e-6)
        for layer_idx in range(2):
            self.assertEqual(
                hf_cache.layers[layer_idx].keys.shape,
                native_cache.layers[layer_idx].keys.shape,
            )
            self.assertLess(
                (
                    hf_cache.layers[layer_idx].keys
                    - native_cache.layers[layer_idx].keys
                ).abs().max().item(),
                1e-6,
            )

    def test_full_causal_lm_with_shared_tail_matches_hf_logits(self) -> None:
        import torch
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
        from wkvm.models.gemma import GemmaRoutedSpanConfig
        from wkvm.runner.gemma_native_forward import NativeGemma4ForCausalLM
        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        torch.manual_seed(31)
        cfg = _tiny_shared_config()
        hf_model = Gemma4ForCausalLM(cfg).eval()
        native_model = NativeGemma4ForCausalLM(hf_model)
        native_cfg = GemmaRoutedSpanConfig(
            num_hidden_layers=cfg.num_hidden_layers,
            num_kv_shared_layers=cfg.num_kv_shared_layers,
            layer_types=tuple(cfg.layer_types),
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            sliding_window=cfg.sliding_window,
        )
        hf_cache = NativeGemmaRoutedCache(cfg, native_cfg)
        native_cache = NativeGemmaRoutedCache(cfg, native_cfg)

        input_ids = torch.tensor([[1, 7, 9, 11, 13]])
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
        mask = _causal_mask(1, input_ids.shape[1], dtype=torch.float32, device=input_ids.device)
        mask_mapping = {"sliding_attention": mask, "full_attention": mask}
        decode_ids = torch.tensor([[17]])
        decode_position_ids = torch.tensor([[input_ids.shape[1]]])

        with torch.inference_mode():
            hf_prefill = hf_model(
                input_ids=input_ids,
                attention_mask=mask_mapping,
                position_ids=position_ids,
                past_key_values=hf_cache,
                use_cache=True,
                logits_to_keep=1,
            )
            native_prefill = native_model(
                input_ids=input_ids,
                attention_mask=mask_mapping,
                position_ids=position_ids,
                past_key_values=native_cache,
                use_cache=True,
                logits_to_keep=1,
            )
            hf_decode = hf_model(
                input_ids=decode_ids,
                attention_mask={"sliding_attention": None, "full_attention": None},
                position_ids=decode_position_ids,
                past_key_values=hf_cache,
                use_cache=True,
                logits_to_keep=1,
            )
            native_decode = native_model(
                input_ids=decode_ids,
                attention_mask={"sliding_attention": None, "full_attention": None},
                position_ids=decode_position_ids,
                past_key_values=native_cache,
                use_cache=True,
                logits_to_keep=1,
            )

        self.assertLess((hf_prefill.logits - native_prefill.logits).abs().max().item(), 5e-6)
        self.assertLess((hf_decode.logits - native_decode.logits).abs().max().item(), 5e-6)
        self.assertEqual(len(hf_cache.layers), 2)
        self.assertEqual(len(native_cache.layers), 2)
        for layer_idx in range(2):
            self.assertEqual(
                hf_cache.layers[layer_idx].keys.shape,
                native_cache.layers[layer_idx].keys.shape,
            )
            self.assertLess(
                (
                    hf_cache.layers[layer_idx].values
                    - native_cache.layers[layer_idx].values
                ).abs().max().item(),
                1e-6,
            )

    def test_causal_lm_bridge_exposes_runner_model_interface(self) -> None:
        import torch
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
        from wkvm.runner.gemma_native_forward import NativeGemma4ForCausalLM

        cfg = _tiny_shared_config()
        hf_model = Gemma4ForCausalLM(cfg)
        native_model = NativeGemma4ForCausalLM(hf_model, native_attention_backend="sdpa")

        self.assertEqual(native_model.wkvm_forward_backend, "wkvm_native_gemma_forward_bridge")
        self.assertTrue(native_model.wkvm_no_hf_transformer_forward)
        self.assertEqual(native_model.native_attention_backend, "sdpa")
        self.assertIs(native_model.eval(), native_model)
        self.assertFalse(native_model.training)
        self.assertIs(native_model.train(True), native_model)
        self.assertTrue(native_model.training)
        self.assertEqual(native_model.device, next(hf_model.parameters()).device)
        self.assertIs(next(native_model.parameters()), next(hf_model.parameters()))
        self.assertIs(native_model.to(torch.float32), native_model)

    def test_checkpoint_state_dict_loader_matches_hf_native_bridge(self) -> None:
        import torch
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4ForCausalLM,
            Gemma4TextDecoderLayer,
        )
        from wkvm.runner.gemma_native_forward import (
            NativeGemma4ForCausalLM,
            native_gemma4_from_checkpoint_state_dict,
        )

        torch.manual_seed(45)
        cfg = _tiny_shared_config()
        hf_model = Gemma4ForCausalLM(cfg).eval()
        live_model = NativeGemma4ForCausalLM(hf_model).eval()
        checkpoint_model = native_gemma4_from_checkpoint_state_dict(
            cfg,
            _checkpoint_layout_state_dict(hf_model),
            prefix="model.language_model",
        )

        self.assertTrue(checkpoint_model.wkvm_no_hf_transformer_forward)
        self.assertTrue(checkpoint_model.wkvm_checkpoint_native_loader)
        self.assertFalse(checkpoint_model.wkvm_uses_hf_model_construction)
        self.assertNotIsInstance(checkpoint_model.hf_model, Gemma4ForCausalLM)
        self.assertFalse(
            any(
                isinstance(layer.hf_layer, Gemma4TextDecoderLayer)
                for layer in checkpoint_model.text_prefix.layers
            )
        )

        input_ids = torch.tensor([[1, 7, 9, 11, 13]])
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
        mask = _causal_mask(1, input_ids.shape[1], dtype=torch.float32, device=input_ids.device)
        mask_mapping = {"sliding_attention": mask, "full_attention": mask}

        with torch.inference_mode():
            live_out = live_model(
                input_ids=input_ids,
                attention_mask=mask_mapping,
                position_ids=position_ids,
                use_cache=False,
                logits_to_keep=1,
            )
            checkpoint_out = checkpoint_model(
                input_ids=input_ids,
                attention_mask=mask_mapping,
                position_ids=position_ids,
                use_cache=False,
                logits_to_keep=1,
            )

        self.assertLess((live_out.logits - checkpoint_out.logits).abs().max().item(), 1e-6)

    def test_checkpoint_state_dict_loader_ties_missing_lm_head(self) -> None:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
        from wkvm.runner.gemma_native_forward import native_gemma4_from_checkpoint_state_dict

        cfg = _tiny_shared_config()
        hf_model = Gemma4ForCausalLM(cfg).eval()
        state = _checkpoint_layout_state_dict(hf_model)
        del state["lm_head.weight"]
        checkpoint_model = native_gemma4_from_checkpoint_state_dict(
            cfg,
            state,
            prefix="model.language_model",
        )

        self.assertTrue(checkpoint_model.hf_model.tie_word_embeddings)
        self.assertEqual(
            checkpoint_model.hf_model.lm_head.weight.data_ptr(),
            checkpoint_model.hf_model.model.embed_tokens.weight.data_ptr(),
        )
        names = [name for name, _tensor in checkpoint_model.hf_model.named_parameters()]
        self.assertNotIn("lm_head.weight", names)

    def test_owned_weight_backend_copies_decoder_layer_tensors(self) -> None:
        import torch
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer
        from wkvm.runner.gemma_native_forward import NativeGemma4TextDecoderLayer

        torch.manual_seed(41)
        cfg = _tiny_config()
        hf_layer = Gemma4TextDecoderLayer(cfg, layer_idx=0).eval()
        native_layer = NativeGemma4TextDecoderLayer(
            hf_layer,
            native_weight_backend="owned",
        )

        self.assertEqual(native_layer.native_weight_backend, "owned")
        self.assertNotEqual(
            native_layer.q_proj.weight.data_ptr(),
            hf_layer.self_attn.q_proj.weight.data_ptr(),
        )
        self.assertTrue(torch.equal(native_layer.q_proj.weight, hf_layer.self_attn.q_proj.weight))

        with torch.no_grad():
            hf_layer.self_attn.q_proj.weight.add_(1.0)
        self.assertFalse(torch.equal(native_layer.q_proj.weight, hf_layer.self_attn.q_proj.weight))

    def test_owned_cpu_weight_backend_keeps_decoder_layer_tensors_on_cpu(self) -> None:
        import torch
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer
        from wkvm.runner.gemma_native_forward import NativeGemma4TextDecoderLayer

        torch.manual_seed(42)
        cfg = _tiny_config()
        hf_layer = Gemma4TextDecoderLayer(cfg, layer_idx=0).eval()
        if torch.cuda.is_available():
            hf_layer = hf_layer.to("cuda")
        native_layer = NativeGemma4TextDecoderLayer(
            hf_layer,
            native_weight_backend="owned_cpu",
        )

        self.assertEqual(native_layer.native_weight_backend, "owned_cpu")
        self.assertEqual(native_layer.q_proj.weight.device.type, "cpu")
        self.assertNotEqual(
            native_layer.q_proj.weight.data_ptr(),
            hf_layer.self_attn.q_proj.weight.data_ptr(),
        )
        self.assertTrue(
            torch.equal(
                native_layer.q_proj.weight,
                hf_layer.self_attn.q_proj.weight.detach().cpu(),
            )
        )

        with torch.no_grad():
            hf_layer.self_attn.q_proj.weight.add_(1.0)
        self.assertFalse(
            torch.equal(
                native_layer.q_proj.weight,
                hf_layer.self_attn.q_proj.weight.detach().cpu(),
            )
        )

    def test_owned_weight_backend_matches_hf_live_bridge(self) -> None:
        import torch
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
        from wkvm.models.gemma import GemmaRoutedSpanConfig
        from wkvm.runner.gemma_native_forward import NativeGemma4ForCausalLM
        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        torch.manual_seed(43)
        cfg = _tiny_shared_config()
        hf_model = Gemma4ForCausalLM(cfg).eval()
        live_model = NativeGemma4ForCausalLM(
            hf_model,
            native_weight_backend="hf_live",
        )
        native_cfg = GemmaRoutedSpanConfig(
            num_hidden_layers=cfg.num_hidden_layers,
            num_kv_shared_layers=cfg.num_kv_shared_layers,
            layer_types=tuple(cfg.layer_types),
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            sliding_window=cfg.sliding_window,
        )
        live_cache = NativeGemmaRoutedCache(cfg, native_cfg)
        owned_cache = NativeGemmaRoutedCache(cfg, native_cfg)

        input_ids = torch.tensor([[1, 7, 9, 11, 13]])
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
        mask = _causal_mask(1, input_ids.shape[1], dtype=torch.float32, device=input_ids.device)
        mask_mapping = {"sliding_attention": mask, "full_attention": mask}

        for backend in ("owned", "owned_cpu"):
            with self.subTest(backend=backend):
                owned_model = NativeGemma4ForCausalLM(
                    hf_model,
                    native_weight_backend=backend,
                )
                live_cache = NativeGemmaRoutedCache(cfg, native_cfg)
                owned_cache = NativeGemmaRoutedCache(cfg, native_cfg)
                with torch.inference_mode():
                    live_out = live_model(
                        input_ids=input_ids,
                        attention_mask=mask_mapping,
                        position_ids=position_ids,
                        past_key_values=live_cache,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                    owned_out = owned_model(
                        input_ids=input_ids,
                        attention_mask=mask_mapping,
                        position_ids=position_ids,
                        past_key_values=owned_cache,
                        use_cache=True,
                        logits_to_keep=1,
                    )

                self.assertLess((live_out.logits - owned_out.logits).abs().max().item(), 1e-6)
                for layer_idx in range(len(live_cache.layers)):
                    self.assertLess(
                        (
                            live_cache.layers[layer_idx].keys
                            - owned_cache.layers[layer_idx].keys
                        ).abs().max().item(),
                        1e-6,
                    )

    def test_owned_release_replaces_hf_decoder_layers_and_matches_live_bridge(self) -> None:
        import torch
        import torch.nn as nn
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
        from wkvm.models.gemma import GemmaRoutedSpanConfig
        from wkvm.runner.gemma_native_forward import NativeGemma4ForCausalLM
        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        torch.manual_seed(44)
        cfg = _tiny_shared_config()
        live_hf_model = Gemma4ForCausalLM(cfg).eval()
        released_hf_model = Gemma4ForCausalLM(cfg).eval()
        released_hf_model.load_state_dict(live_hf_model.state_dict())

        live_model = NativeGemma4ForCausalLM(
            live_hf_model,
            native_weight_backend="hf_live",
        )
        released_model = NativeGemma4ForCausalLM(
            released_hf_model,
            native_weight_backend="owned",
            release_hf_decoder_layers=True,
        )
        self.assertTrue(released_model.release_hf_decoder_layers)
        self.assertEqual(released_model.released_hf_decoder_layers, cfg.num_hidden_layers)
        self.assertTrue(
            all(isinstance(layer, nn.Identity) for layer in released_hf_model.model.layers)
        )
        self.assertTrue(
            all(layer.hf_layer is None for layer in released_model.text_prefix.layers)
        )

        native_cfg = GemmaRoutedSpanConfig(
            num_hidden_layers=cfg.num_hidden_layers,
            num_kv_shared_layers=cfg.num_kv_shared_layers,
            layer_types=tuple(cfg.layer_types),
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            sliding_window=cfg.sliding_window,
        )
        live_cache = NativeGemmaRoutedCache(cfg, native_cfg)
        released_cache = NativeGemmaRoutedCache(cfg, native_cfg)
        input_ids = torch.tensor([[1, 7, 9, 11, 13]])
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
        mask = _causal_mask(1, input_ids.shape[1], dtype=torch.float32, device=input_ids.device)
        mask_mapping = {"sliding_attention": mask, "full_attention": mask}

        with torch.inference_mode():
            live_out = live_model(
                input_ids=input_ids,
                attention_mask=mask_mapping,
                position_ids=position_ids,
                past_key_values=live_cache,
                use_cache=True,
                logits_to_keep=1,
            )
            released_out = released_model(
                input_ids=input_ids,
                attention_mask=mask_mapping,
                position_ids=position_ids,
                past_key_values=released_cache,
                use_cache=True,
                logits_to_keep=1,
            )

        self.assertLess((live_out.logits - released_out.logits).abs().max().item(), 1e-6)
        for layer_idx in range(len(live_cache.layers)):
            self.assertLess(
                (
                    live_cache.layers[layer_idx].keys
                    - released_cache.layers[layer_idx].keys
                ).abs().max().item(),
                1e-6,
            )

    def test_release_hf_decoder_layers_requires_owned_weight_backend(self) -> None:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
        from wkvm.runner.gemma_native_forward import NativeGemma4ForCausalLM

        cfg = _tiny_shared_config()
        hf_model = Gemma4ForCausalLM(cfg).eval()
        with self.assertRaisesRegex(ValueError, "requires native_weight_backend"):
            NativeGemma4ForCausalLM(
                hf_model,
                native_weight_backend="hf_live",
                release_hf_decoder_layers=True,
            )

    def test_packed_projection_backends_match_separate_layer(self) -> None:
        import torch
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextDecoderLayer,
            Gemma4TextRotaryEmbedding,
        )
        from wkvm.runner.gemma_native_forward import NativeGemma4TextDecoderLayer

        torch.manual_seed(47)
        cfg = _tiny_config()
        hf_layer = Gemma4TextDecoderLayer(cfg, layer_idx=0).eval()
        rotary = Gemma4TextRotaryEmbedding(cfg)

        hidden = torch.randn(2, 5, cfg.hidden_size)
        per_layer_input = torch.randn(2, 5, cfg.hidden_size_per_layer_input)
        position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)
        position_embeddings = rotary(hidden, position_ids, "sliding_attention")
        mask = _causal_mask(2, 5, dtype=hidden.dtype, device=hidden.device)

        for weight_backend in ("hf_live", "owned", "owned_cpu"):
            for projection_backend in (
                "qkv_packed",
                "gate_up_packed",
                "qkv_gate_up_packed",
            ):
                with self.subTest(
                    weight_backend=weight_backend,
                    projection_backend=projection_backend,
                ):
                    separate_layer = NativeGemma4TextDecoderLayer(
                        hf_layer,
                        native_projection_backend="separate",
                        native_weight_backend=weight_backend,
                    )
                    packed_layer = NativeGemma4TextDecoderLayer(
                        hf_layer,
                        native_projection_backend=projection_backend,
                        native_weight_backend=weight_backend,
                    )

                    with torch.inference_mode():
                        expected = separate_layer(
                            hidden,
                            per_layer_input,
                            shared_kv_states=UserDict(),
                            position_embeddings=position_embeddings,
                            attention_mask=mask,
                            position_ids=position_ids,
                            past_key_values=None,
                        )
                        actual = packed_layer(
                            hidden,
                            per_layer_input,
                            shared_kv_states=UserDict(),
                            position_embeddings=position_embeddings,
                            attention_mask=mask,
                            position_ids=position_ids,
                            past_key_values=None,
                        )

                    self.assertLess((expected - actual).abs().max().item(), 2e-6)

    def test_projection_backend_alias_packed_uses_qkv_and_gate_up(self) -> None:
        import torch
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer
        from wkvm.runner.gemma_native_forward import NativeGemma4TextDecoderLayer

        cfg = _tiny_config()
        hf_layer = Gemma4TextDecoderLayer(cfg, layer_idx=0).eval()
        native_layer = NativeGemma4TextDecoderLayer(
            hf_layer,
            native_projection_backend="packed",
        )

        self.assertEqual(native_layer.native_projection_backend, "qkv_gate_up_packed")
        self.assertIsNotNone(native_layer._qkv_proj)
        self.assertIsNotNone(native_layer._gate_up_proj)

    def test_owned_packed_projection_drops_replaced_individual_snapshots(self) -> None:
        from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer
        from wkvm.runner.gemma_native_forward import NativeGemma4TextDecoderLayer

        cfg = _tiny_config()
        hf_layer = Gemma4TextDecoderLayer(cfg, layer_idx=0).eval()
        native_layer = NativeGemma4TextDecoderLayer(
            hf_layer,
            native_projection_backend="qkv_gate_up_packed",
            native_weight_backend="owned",
        )

        self.assertIsNotNone(native_layer._qkv_proj)
        self.assertIsNone(native_layer.q_proj)
        self.assertIsNone(native_layer.k_proj)
        self.assertIsNone(native_layer.v_proj)
        self.assertIsNotNone(native_layer._gate_up_proj)
        self.assertIsNone(native_layer.mlp_gate_proj)
        self.assertIsNone(native_layer.mlp_up_proj)
        self.assertIsNotNone(native_layer.o_proj)
        self.assertIsNotNone(native_layer.mlp_down_proj)

    def test_qkv_and_gate_up_packed_layer_matches_separate_layer(self) -> None:
        import torch
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextDecoderLayer,
            Gemma4TextRotaryEmbedding,
        )
        from wkvm.runner.gemma_native_forward import NativeGemma4TextDecoderLayer

        torch.manual_seed(49)
        cfg = _tiny_config()
        hf_layer = Gemma4TextDecoderLayer(cfg, layer_idx=0).eval()
        rotary = Gemma4TextRotaryEmbedding(cfg)

        hidden = torch.randn(2, 5, cfg.hidden_size)
        per_layer_input = torch.randn(2, 5, cfg.hidden_size_per_layer_input)
        position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)
        position_embeddings = rotary(hidden, position_ids, "sliding_attention")
        mask = _causal_mask(2, 5, dtype=hidden.dtype, device=hidden.device)

        separate_layer = NativeGemma4TextDecoderLayer(
            hf_layer,
            native_projection_backend="separate",
        )
        packed_layer = NativeGemma4TextDecoderLayer(
            hf_layer,
            native_projection_backend="qkv_gate_up_packed",
        )

        with torch.inference_mode():
            expected = separate_layer(
                hidden,
                per_layer_input,
                shared_kv_states=UserDict(),
                position_embeddings=position_embeddings,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_values=None,
            )
            actual = packed_layer(
                hidden,
                per_layer_input,
                shared_kv_states=UserDict(),
                position_embeddings=position_embeddings,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_values=None,
            )

        self.assertLess((expected - actual).abs().max().item(), 2e-6)

    def test_owned_qkv_packed_layer_matches_owned_separate_layer(self) -> None:
        import torch
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4TextDecoderLayer,
            Gemma4TextRotaryEmbedding,
        )
        from wkvm.runner.gemma_native_forward import NativeGemma4TextDecoderLayer

        torch.manual_seed(47)
        cfg = _tiny_config()
        hf_layer = Gemma4TextDecoderLayer(cfg, layer_idx=0).eval()
        rotary = Gemma4TextRotaryEmbedding(cfg)

        hidden = torch.randn(2, 5, cfg.hidden_size)
        per_layer_input = torch.randn(2, 5, cfg.hidden_size_per_layer_input)
        position_ids = torch.arange(5).unsqueeze(0).expand(2, -1)
        position_embeddings = rotary(hidden, position_ids, "sliding_attention")
        mask = _causal_mask(2, 5, dtype=hidden.dtype, device=hidden.device)

        for backend in ("owned", "owned_cpu"):
            with self.subTest(backend=backend):
                separate_layer = NativeGemma4TextDecoderLayer(
                    hf_layer,
                    native_projection_backend="separate",
                    native_weight_backend=backend,
                )
                packed_layer = NativeGemma4TextDecoderLayer(
                    hf_layer,
                    native_projection_backend="qkv_packed",
                    native_weight_backend=backend,
                )

                with torch.inference_mode():
                    expected = separate_layer(
                        hidden,
                        per_layer_input,
                        shared_kv_states=UserDict(),
                        position_embeddings=position_embeddings,
                        attention_mask=mask,
                        position_ids=position_ids,
                        past_key_values=None,
                    )
                    actual = packed_layer(
                        hidden,
                        per_layer_input,
                        shared_kv_states=UserDict(),
                        position_embeddings=position_embeddings,
                        attention_mask=mask,
                        position_ids=position_ids,
                        past_key_values=None,
                    )

                self.assertLess((expected - actual).abs().max().item(), 2e-6)

    def test_causal_lm_sdpa_backend_matches_manual_backend(self) -> None:
        import torch
        from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
        from wkvm.models.gemma import GemmaRoutedSpanConfig
        from wkvm.runner.gemma_native_forward import NativeGemma4ForCausalLM
        from wkvm.runner.gemma_runner import NativeGemmaRoutedCache

        torch.manual_seed(37)
        cfg = _tiny_shared_config()
        hf_model = Gemma4ForCausalLM(cfg).eval()
        manual_model = NativeGemma4ForCausalLM(
            hf_model,
            native_attention_backend="manual",
        )
        sdpa_model = NativeGemma4ForCausalLM(
            hf_model,
            native_attention_backend="sdpa",
        )
        native_cfg = GemmaRoutedSpanConfig(
            num_hidden_layers=cfg.num_hidden_layers,
            num_kv_shared_layers=cfg.num_kv_shared_layers,
            layer_types=tuple(cfg.layer_types),
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            sliding_window=cfg.sliding_window,
        )
        manual_cache = NativeGemmaRoutedCache(cfg, native_cfg)
        sdpa_cache = NativeGemmaRoutedCache(cfg, native_cfg)

        input_ids = torch.tensor([[1, 7, 9, 11, 13]])
        position_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
        mask = _causal_mask(1, input_ids.shape[1], dtype=torch.float32, device=input_ids.device)
        mask_mapping = {"sliding_attention": mask, "full_attention": mask}

        with torch.inference_mode():
            manual_out = manual_model(
                input_ids=input_ids,
                attention_mask=mask_mapping,
                position_ids=position_ids,
                past_key_values=manual_cache,
                use_cache=True,
                logits_to_keep=1,
            )
            sdpa_out = sdpa_model(
                input_ids=input_ids,
                attention_mask=mask_mapping,
                position_ids=position_ids,
                past_key_values=sdpa_cache,
                use_cache=True,
                logits_to_keep=1,
            )

        self.assertLess((manual_out.logits - sdpa_out.logits).abs().max().item(), 1e-5)


if __name__ == "__main__":
    unittest.main()
