import unittest


class TestGemmaTokenPool(unittest.TestCase):
    def test_dense_padded_triton_decode_matches_manual_gqa(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        try:
            from wkvm.runner.gemma_token_pool_triton import dense_padded_gqa_decode
        except ImportError:
            self.skipTest("triton unavailable")
        from types import SimpleNamespace
        from wkvm.runner.gemma_native_forward import _attention_forward_manual_gqa

        torch.manual_seed(31)
        attn = SimpleNamespace(
            num_key_value_groups=2,
            scaling=0.25,
            attention_dropout=0.0,
            training=False,
        )
        query = torch.randn(2, 4, 1, 16, device="cuda")
        keys = torch.randn(2, 2, 7, 16, device="cuda")
        values = torch.randn(2, 2, 7, 16, device="cuda")
        mask = torch.zeros(2, 1, 1, 7, device="cuda")
        mask[0, :, :, 5:] = torch.finfo(mask.dtype).min
        mask[1, :, :, 6:] = torch.finfo(mask.dtype).min

        actual = dense_padded_gqa_decode(
            query,
            keys,
            values,
            mask,
            num_key_value_groups=2,
            scaling=attn.scaling,
        )
        expected, _ = _attention_forward_manual_gqa(
            attn,
            query,
            keys,
            values,
            mask,
        )
        torch.cuda.synchronize()

        self.assertLess((expected - actual).abs().max().item(), 1e-4)

    def test_triton_block_n_env_selection_prefers_paged_override(self) -> None:
        import os

        try:
            from wkvm.runner.gemma_token_pool_triton import (
                _resolve_block_n,
                _resolve_num_warps,
            )
        except ImportError:
            self.skipTest("triton unavailable")

        old_common = os.environ.get("WKVM_TOKEN_POOL_TRITON_BLOCK_N")
        old_paged = os.environ.get("WKVM_TOKEN_POOL_PAGED_TRITON_BLOCK_N")
        old_warps = os.environ.get("WKVM_TOKEN_POOL_TRITON_NUM_WARPS")
        old_paged_warps = os.environ.get("WKVM_TOKEN_POOL_PAGED_TRITON_NUM_WARPS")
        try:
            os.environ.pop("WKVM_TOKEN_POOL_TRITON_BLOCK_N", None)
            os.environ.pop("WKVM_TOKEN_POOL_PAGED_TRITON_BLOCK_N", None)
            os.environ.pop("WKVM_TOKEN_POOL_TRITON_NUM_WARPS", None)
            os.environ.pop("WKVM_TOKEN_POOL_PAGED_TRITON_NUM_WARPS", None)
            self.assertEqual(
                _resolve_block_n(
                    256,
                    None,
                    env_names=("WKVM_TOKEN_POOL_TRITON_BLOCK_N",),
                ),
                32,
            )
            self.assertEqual(
                _resolve_block_n(
                    512,
                    None,
                    env_names=("WKVM_TOKEN_POOL_TRITON_BLOCK_N",),
                ),
                32,
            )

            os.environ["WKVM_TOKEN_POOL_TRITON_BLOCK_N"] = "16"
            self.assertEqual(
                _resolve_block_n(
                    512,
                    None,
                    env_names=(
                        "WKVM_TOKEN_POOL_PAGED_TRITON_BLOCK_N",
                        "WKVM_TOKEN_POOL_TRITON_BLOCK_N",
                    ),
                ),
                16,
            )

            os.environ["WKVM_TOKEN_POOL_PAGED_TRITON_BLOCK_N"] = "8"
            self.assertEqual(
                _resolve_block_n(
                    512,
                    None,
                    env_names=(
                        "WKVM_TOKEN_POOL_PAGED_TRITON_BLOCK_N",
                        "WKVM_TOKEN_POOL_TRITON_BLOCK_N",
                    ),
                ),
                8,
            )
            self.assertEqual(
                _resolve_block_n(
                    512,
                    4,
                    env_names=(
                        "WKVM_TOKEN_POOL_PAGED_TRITON_BLOCK_N",
                        "WKVM_TOKEN_POOL_TRITON_BLOCK_N",
                    ),
                ),
                4,
            )
            with self.assertRaisesRegex(ValueError, "block_n"):
                _resolve_block_n(
                    512,
                    0,
                    env_names=("WKVM_TOKEN_POOL_TRITON_BLOCK_N",),
                )

            self.assertEqual(
                _resolve_num_warps(
                    256,
                    env_names=("WKVM_TOKEN_POOL_TRITON_NUM_WARPS",),
                ),
                4,
            )
            self.assertEqual(
                _resolve_num_warps(
                    512,
                    env_names=("WKVM_TOKEN_POOL_TRITON_NUM_WARPS",),
                ),
                4,
            )
            self.assertEqual(
                _resolve_num_warps(
                    1024,
                    env_names=("WKVM_TOKEN_POOL_TRITON_NUM_WARPS",),
                ),
                8,
            )
            os.environ["WKVM_TOKEN_POOL_TRITON_NUM_WARPS"] = "2"
            self.assertEqual(
                _resolve_num_warps(
                    256,
                    env_names=(
                        "WKVM_TOKEN_POOL_PAGED_TRITON_NUM_WARPS",
                        "WKVM_TOKEN_POOL_TRITON_NUM_WARPS",
                    ),
                ),
                2,
            )
            os.environ["WKVM_TOKEN_POOL_PAGED_TRITON_NUM_WARPS"] = "1"
            self.assertEqual(
                _resolve_num_warps(
                    256,
                    env_names=(
                        "WKVM_TOKEN_POOL_PAGED_TRITON_NUM_WARPS",
                        "WKVM_TOKEN_POOL_TRITON_NUM_WARPS",
                    ),
                ),
                1,
            )
            with self.assertRaisesRegex(ValueError, "num_warps"):
                os.environ["WKVM_TOKEN_POOL_TRITON_NUM_WARPS"] = "3"
                _resolve_num_warps(
                    256,
                    env_names=("WKVM_TOKEN_POOL_TRITON_NUM_WARPS",),
                )
        finally:
            if old_common is None:
                os.environ.pop("WKVM_TOKEN_POOL_TRITON_BLOCK_N", None)
            else:
                os.environ["WKVM_TOKEN_POOL_TRITON_BLOCK_N"] = old_common
            if old_paged is None:
                os.environ.pop("WKVM_TOKEN_POOL_PAGED_TRITON_BLOCK_N", None)
            else:
                os.environ["WKVM_TOKEN_POOL_PAGED_TRITON_BLOCK_N"] = old_paged
            if old_warps is None:
                os.environ.pop("WKVM_TOKEN_POOL_TRITON_NUM_WARPS", None)
            else:
                os.environ["WKVM_TOKEN_POOL_TRITON_NUM_WARPS"] = old_warps
            if old_paged_warps is None:
                os.environ.pop("WKVM_TOKEN_POOL_PAGED_TRITON_NUM_WARPS", None)
            else:
                os.environ["WKVM_TOKEN_POOL_PAGED_TRITON_NUM_WARPS"] = old_paged_warps

    def test_triton_input_precision_defaults_and_env_override(self) -> None:
        import os

        try:
            import torch
            from wkvm.runner.gemma_token_pool_triton import (
                _resolve_input_precision,
                _resolve_native_dot,
            )
        except ImportError:
            self.skipTest("torch/triton unavailable")

        old_precision = os.environ.get("WKVM_TOKEN_POOL_TRITON_INPUT_PRECISION")
        old_dot_dtype = os.environ.get("WKVM_TOKEN_POOL_TRITON_DOT_DTYPE")
        try:
            os.environ.pop("WKVM_TOKEN_POOL_TRITON_INPUT_PRECISION", None)
            os.environ.pop("WKVM_TOKEN_POOL_TRITON_DOT_DTYPE", None)
            self.assertEqual(_resolve_input_precision(torch.float32), "ieee")
            self.assertEqual(_resolve_input_precision(torch.float16), "tf32")
            self.assertEqual(_resolve_input_precision(torch.bfloat16), "tf32")
            self.assertFalse(_resolve_native_dot(torch.float32))
            self.assertTrue(_resolve_native_dot(torch.float16))
            self.assertTrue(_resolve_native_dot(torch.bfloat16))

            os.environ["WKVM_TOKEN_POOL_TRITON_INPUT_PRECISION"] = "tf32x3"
            self.assertEqual(_resolve_input_precision(torch.float32), "tf32x3")
            self.assertEqual(_resolve_input_precision(torch.bfloat16), "tf32x3")
            self.assertEqual(_resolve_input_precision(torch.float32, "ieee"), "ieee")

            os.environ["WKVM_TOKEN_POOL_TRITON_DOT_DTYPE"] = "fp32"
            self.assertFalse(_resolve_native_dot(torch.bfloat16))
            os.environ["WKVM_TOKEN_POOL_TRITON_DOT_DTYPE"] = "native"
            self.assertTrue(_resolve_native_dot(torch.float32))
            self.assertFalse(_resolve_native_dot(torch.float16, "fp32"))

            os.environ["WKVM_TOKEN_POOL_TRITON_INPUT_PRECISION"] = "auto"
            self.assertEqual(_resolve_input_precision(torch.float32), "ieee")
            self.assertEqual(_resolve_input_precision(torch.float16), "tf32")

            with self.assertRaisesRegex(ValueError, "INPUT_PRECISION"):
                _resolve_input_precision(torch.float16, "invalid")
            with self.assertRaisesRegex(ValueError, "DOT_DTYPE"):
                _resolve_native_dot(torch.float16, "invalid")
        finally:
            if old_precision is None:
                os.environ.pop("WKVM_TOKEN_POOL_TRITON_INPUT_PRECISION", None)
            else:
                os.environ["WKVM_TOKEN_POOL_TRITON_INPUT_PRECISION"] = old_precision
            if old_dot_dtype is None:
                os.environ.pop("WKVM_TOKEN_POOL_TRITON_DOT_DTYPE", None)
            else:
                os.environ["WKVM_TOKEN_POOL_TRITON_DOT_DTYPE"] = old_dot_dtype

    def test_token_pool_triton_bfloat16_native_dot_matches_reference(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")

        from wkvm.runner.gemma_token_pool_triton import token_pool_gqa_decode

        torch.manual_seed(1217)
        device = torch.device("cuda")
        batch = 2
        query_heads = 4
        kv_heads = 2
        groups = query_heads // kv_heads
        head_dim = 32
        scaling = head_dim ** -0.5
        query = torch.randn(
            batch,
            query_heads,
            1,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        keys = torch.randn(
            16,
            kv_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        values = torch.randn_like(keys)
        kv_indices = torch.tensor(
            [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11],
            dtype=torch.int32,
            device=device,
        )
        kv_indptr = torch.tensor([0, 6, 12], dtype=torch.int32, device=device)

        actual = token_pool_gqa_decode(
            query,
            keys,
            values,
            kv_indptr,
            kv_indices,
            num_key_value_groups=groups,
            scaling=scaling,
            block_n=16,
        )
        torch.cuda.synchronize()

        expected_rows = []
        qf = query.float()
        kf = keys.float()
        vf = values.float()
        for row, token_indices in enumerate(([0, 2, 4, 6, 8, 10], [1, 3, 5, 7, 9, 11])):
            row_keys = kf[torch.tensor(token_indices, dtype=torch.long, device=device)]
            row_values = vf[torch.tensor(token_indices, dtype=torch.long, device=device)]
            row_outputs = []
            for q_head in range(query_heads):
                kv_head = q_head // groups
                scores = (
                    row_keys[:, kv_head, :] * qf[row, q_head, 0, :].unsqueeze(0)
                ).sum(dim=-1) * scaling
                probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
                row_outputs.append((probs.unsqueeze(0) @ row_values[:, kv_head, :]).squeeze(0))
            expected_rows.append(torch.stack(row_outputs, dim=0))
        expected = torch.stack(expected_rows, dim=0).unsqueeze(1)

        self.assertLess((expected - actual.float()).abs().max().item(), 2e-2)

    def test_split_kv_triton_bfloat16_native_dot_accepts_workspace(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")

        from wkvm.runner.gemma_token_pool_triton import (
            _block_g,
            _resolve_native_dot,
            token_pool_gqa_decode,
            token_pool_gqa_decode_split_kv,
        )

        torch.manual_seed(1218)
        device = torch.device("cuda")
        batch = 2
        query_heads = 4
        kv_heads = 2
        groups = query_heads // kv_heads
        head_dim = 32
        scaling = head_dim ** -0.5
        query = torch.randn(
            batch,
            query_heads,
            1,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        keys = torch.randn(
            128,
            kv_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        values = torch.randn_like(keys)
        row0 = torch.arange(0, 40, dtype=torch.int32, device=device)
        row1 = torch.arange(40, 88, dtype=torch.int32, device=device)
        kv_indices = torch.cat([row0, row1], dim=0)
        kv_indptr = torch.tensor([0, 40, 88], dtype=torch.int32, device=device)
        max_seq_len = 48
        split_size = 16
        max_splits = 3
        block_groups = _block_g(groups, _resolve_native_dot(torch.bfloat16))
        workspace = (
            torch.empty(
                (batch, kv_heads, max_splits, block_groups),
                dtype=torch.float32,
                device=device,
            ),
            torch.empty(
                (batch, kv_heads, max_splits, block_groups),
                dtype=torch.float32,
                device=device,
            ),
            torch.empty(
                (batch, kv_heads, max_splits, block_groups, head_dim),
                dtype=torch.float32,
                device=device,
            ),
        )

        expected = token_pool_gqa_decode(
            query,
            keys,
            values,
            kv_indptr,
            kv_indices,
            num_key_value_groups=groups,
            scaling=scaling,
            block_n=16,
        )
        actual = token_pool_gqa_decode_split_kv(
            query,
            keys,
            values,
            kv_indptr,
            kv_indices,
            num_key_value_groups=groups,
            scaling=scaling,
            max_seq_len=max_seq_len,
            split_size=split_size,
            min_splits=2,
            block_n=16,
            workspace=workspace,
        )
        torch.cuda.synchronize()

        self.assertLess((expected.float() - actual.float()).abs().max().item(), 2e-2)

    def test_req_to_token_table_alloc_append_metadata_and_free(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import ReqToTokenTable

        table = ReqToTokenTable(max_requests=2, max_context_len=8)
        a = table.allocate("a")
        b = table.allocate("b")
        self.assertEqual((a, b), (0, 1))
        with self.assertRaises(RuntimeError):
            table.allocate("overflow")

        table.append_slots(a, [10, 11, 12, 13])
        table.append_slots(b, torch.tensor([20, 21, 22], dtype=torch.int32))
        self.assertEqual(table.length("a"), 4)
        self.assertEqual(table.length(b), 3)
        self.assertEqual(table.slots_for("a").tolist(), [10, 11, 12, 13])

        full = table.build_decode_metadata(
            [a, b],
            out_cache_loc=[13, 22],
        )
        self.assertEqual(full.req_pool_indices.tolist(), [0, 1])
        self.assertEqual(full.logical_seq_lens.tolist(), [4, 3])
        self.assertEqual(full.seq_lens.tolist(), [4, 3])
        self.assertEqual(full.kv_indptr.tolist(), [0, 4, 7])
        self.assertEqual(full.kv_indices.tolist(), [10, 11, 12, 13, 20, 21, 22])
        self.assertEqual(full.out_cache_loc.tolist(), [13, 22])
        self.assertEqual(full.out_cache_loc_long.tolist(), [13, 22])
        self.assertEqual(full.out_cache_loc_long.dtype, torch.long)

        sliding = table.build_decode_metadata(
            [a, b],
            out_cache_loc=[13, 22],
            sliding_window=2,
        )
        self.assertEqual(sliding.logical_seq_lens.tolist(), [4, 3])
        self.assertEqual(sliding.seq_lens.tolist(), [2, 2])
        self.assertEqual(sliding.kv_indptr.tolist(), [0, 2, 4])
        self.assertEqual(sliding.kv_indices.tolist(), [12, 13, 21, 22])

        paged_table = ReqToTokenTable(max_requests=2, max_context_len=8)
        pa = paged_table.allocate("pa")
        pb = paged_table.allocate("pb")
        paged_table.append_slots(pa, [0, 1, 2, 3])
        paged_table.append_slots(pb, [8, 9, 10])
        paged = paged_table.build_paged_decode_metadata(
            [pa, pb],
            block_size=4,
            out_cache_loc=[3, 10],
            sliding_window=2,
            token_pool_capacity=32,
        )
        self.assertEqual(paged.req_pool_indices.tolist(), [0, 1])
        self.assertEqual(paged.logical_seq_lens.tolist(), [4, 3])
        self.assertEqual(paged.seq_lens.tolist(), [2, 2])
        self.assertEqual(paged.selected_start_positions.tolist(), [2, 1])
        self.assertEqual(paged.block_tables.tolist(), [[0], [2]])
        self.assertEqual(paged.block_table_lens.tolist(), [1, 1])
        self.assertEqual(paged.slot_mapping.tolist(), [3, 10])
        self.assertEqual(paged.out_cache_loc_long.tolist(), [3, 10])

        table.free("a")
        c = table.allocate("c")
        self.assertEqual(c, a)
        self.assertEqual(table.length(c), 0)
        self.assertTrue((table.req_to_token[c] == table.padding_token).all().item())

    def test_req_to_token_table_clear_before_tracks_new_prefix_only(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import ReqToTokenTable

        table = ReqToTokenTable(max_requests=1, max_context_len=8)
        slot = table.allocate("a")
        table.append_slots(slot, range(8))

        self.assertEqual(table.clear_before(slot, 4), [0, 1, 2, 3])
        self.assertEqual(table.clear_before(slot, 6), [4, 5])
        self.assertEqual(table.clear_before(slot, 6), [])
        self.assertEqual(
            table.slots_for(slot).tolist(),
            [-1, -1, -1, -1, -1, -1, 6, 7],
        )

        table.truncate(slot, 2)
        table.append_slots(slot, [8, 9])
        self.assertEqual(table.slots_for(slot).tolist(), [-1, -1, 8, 9])
        self.assertEqual(table.clear_before(slot, 4), [8, 9])

        table.free("a")
        reused = table.allocate("reused")
        self.assertEqual(reused, slot)
        table.append_slots(reused, [10, 11])
        self.assertEqual(table.clear_before("reused", 1), [10])

    def test_req_to_token_table_reuses_decode_metadata_workspace_by_key(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenPoolDecodeMetadataWorkspace,
        )

        table = ReqToTokenTable(max_requests=2, max_context_len=8)
        self.assertIsInstance(
            table.decode_metadata_workspace,
            TokenPoolDecodeMetadataWorkspace,
        )
        self.assertIs(
            table._decode_metadata_workspaces,
            table.decode_metadata_workspace.flat_workspaces,
        )
        a = table.allocate("a")
        b = table.allocate("b")
        table.append_slots(a, [10, 11, 12, 13])
        table.append_slots(b, torch.tensor([20, 21, 22, 23], dtype=torch.int32))

        first = table.build_decode_metadata(
            [a, b],
            out_cache_loc=[13, 23],
            workspace_key="full_attention",
        )
        first_ptr = int(first.kv_indices.data_ptr())
        second = table.build_decode_metadata(
            [a, b],
            out_cache_loc=[13, 23],
            workspace_key="full_attention",
        )
        self.assertEqual(int(second.kv_indices.data_ptr()), first_ptr)
        self.assertEqual(second.kv_indices.tolist(), [10, 11, 12, 13, 20, 21, 22, 23])
        self.assertEqual(second.kv_indptr.tolist(), [0, 4, 8])
        self.assertEqual(second.out_cache_loc_long.dtype, torch.long)

        sliding = table.build_decode_metadata(
            [a, b],
            out_cache_loc=[13, 23],
            sliding_window=2,
            workspace_key="sliding_attention",
        )
        self.assertNotEqual(int(sliding.kv_indices.data_ptr()), first_ptr)
        self.assertEqual(sliding.kv_indices.tolist(), [12, 13, 22, 23])
        self.assertEqual(sliding.kv_indptr.tolist(), [0, 2, 4])
        self.assertEqual(second.kv_indices.tolist(), [10, 11, 12, 13, 20, 21, 22, 23])

    def test_req_to_token_table_reuses_paged_decode_metadata_workspace_by_key(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenPoolDecodeMetadataWorkspace,
        )

        table = ReqToTokenTable(max_requests=2, max_context_len=8)
        self.assertIsInstance(
            table.decode_metadata_workspace,
            TokenPoolDecodeMetadataWorkspace,
        )
        self.assertIs(
            table._paged_decode_metadata_workspaces,
            table.decode_metadata_workspace.paged_workspaces,
        )
        a = table.allocate("a")
        b = table.allocate("b")
        table.append_slots(a, [0, 1, 2, 3])
        table.append_slots(b, torch.tensor([8, 9, 10, 11], dtype=torch.int32))

        first = table.build_paged_decode_metadata(
            [a, b],
            block_size=4,
            block_table_width=2,
            out_cache_loc=[3, 11],
            sliding_window=2,
            token_pool_capacity=32,
            workspace_key="sliding_attention_paged",
        )
        block_ptr = int(first.block_tables.data_ptr())
        start_ptr = int(first.selected_start_positions.data_ptr())
        second = table.build_paged_decode_metadata(
            [a, b],
            block_size=4,
            block_table_width=2,
            out_cache_loc=[3, 11],
            sliding_window=2,
            token_pool_capacity=32,
            workspace_key="sliding_attention_paged",
        )
        self.assertEqual(int(second.block_tables.data_ptr()), block_ptr)
        self.assertEqual(int(second.selected_start_positions.data_ptr()), start_ptr)
        self.assertEqual(second.block_tables.tolist(), [[0, -1], [2, -1]])
        self.assertEqual(second.block_table_lens.tolist(), [1, 1])
        self.assertEqual(second.selected_start_positions.tolist(), [2, 2])
        self.assertEqual(second.slot_mapping.tolist(), [3, 11])
        self.assertEqual(second.slot_mapping.dtype, torch.long)

    def test_req_to_token_table_builds_paged_metadata_from_page_tables(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import ReqToTokenTable

        table = ReqToTokenTable(max_requests=2, max_context_len=40)
        a = table.allocate("a")
        b = table.allocate("b")
        table.append_slots(a, torch.arange(18, dtype=torch.int32))
        table.append_slots(b, torch.arange(32, 50, dtype=torch.int32))

        metadata = table.build_paged_decode_metadata_from_page_tables(
            [a, b],
            [{0: 0, 1: 1, 2: 2}, {3: 8, 4: 9, 5: 10}],
            block_size=4,
            block_table_width=4,
            seq_lens=[10, 18],
            out_cache_loc=[9, 37],
            sliding_window=6,
            token_pool_capacity=64,
            workspace_key="sliding_attention_paged",
        )

        self.assertEqual(metadata.req_pool_indices.tolist(), [0, 1])
        self.assertEqual(metadata.logical_seq_lens.tolist(), [10, 18])
        self.assertEqual(metadata.seq_lens.tolist(), [6, 6])
        self.assertEqual(metadata.selected_start_positions.tolist(), [4, 12])
        self.assertEqual(metadata.block_tables.tolist(), [[1, 2, -1, -1], [8, 9, -1, -1]])
        self.assertEqual(metadata.block_table_lens.tolist(), [2, 2])
        self.assertEqual(metadata.out_cache_loc.tolist(), [9, 37])
        self.assertEqual(metadata.slot_mapping.dtype, torch.long)

        dense_page_table = torch.full((2, 6), -1, dtype=torch.int32)
        dense_page_table[a, 0] = 0
        dense_page_table[a, 1] = 1
        dense_page_table[a, 2] = 2
        dense_page_table[b, 3] = 8
        dense_page_table[b, 4] = 9
        dense_page_table[b, 5] = 10
        dense_metadata = table.build_paged_decode_metadata_from_page_table_tensor(
            [a, b],
            dense_page_table,
            block_size=4,
            block_table_width=4,
            seq_lens=[10, 18],
            out_cache_loc=[9, 37],
            sliding_window=6,
            token_pool_capacity=64,
            workspace_key="sliding_attention_paged_tensor",
        )

        self.assertEqual(dense_metadata.req_pool_indices.tolist(), [0, 1])
        self.assertEqual(dense_metadata.logical_seq_lens.tolist(), [10, 18])
        self.assertEqual(dense_metadata.seq_lens.tolist(), [6, 6])
        self.assertEqual(dense_metadata.selected_start_positions.tolist(), [4, 12])
        self.assertEqual(
            dense_metadata.block_tables.tolist(),
            [[1, 2, -1, -1], [8, 9, -1, -1]],
        )
        self.assertEqual(dense_metadata.block_table_lens.tolist(), [2, 2])
        self.assertEqual(dense_metadata.out_cache_loc.tolist(), [9, 37])
        self.assertEqual(dense_metadata.slot_mapping.dtype, torch.long)
        dense_block_ptr = int(dense_metadata.block_tables.data_ptr())
        dense_again = table.build_paged_decode_metadata_from_page_table_tensor(
            [a, b],
            dense_page_table,
            block_size=4,
            block_table_width=4,
            seq_lens=[10, 18],
            out_cache_loc=[9, 37],
            sliding_window=6,
            token_pool_capacity=64,
            workspace_key="sliding_attention_paged_tensor",
        )
        self.assertEqual(int(dense_again.block_tables.data_ptr()), dense_block_ptr)
        self.assertIs(
            table._paged_decode_metadata_workspaces["sliding_attention_paged_tensor"],
            table.decode_metadata_workspace.paged_workspace(
                "sliding_attention_paged_tensor"
            ),
        )

        dense_page_table[a, 2] = -1
        with self.assertRaisesRegex(ValueError, "missing"):
            table.build_paged_decode_metadata_from_page_table_tensor(
                [a],
                dense_page_table,
                block_size=4,
                seq_lens=[10],
                out_cache_loc=[9],
                sliding_window=6,
            )

        with self.assertRaisesRegex(ValueError, "final logical token"):
            table.build_paged_decode_metadata_from_page_tables(
                [a],
                [{0: 0, 1: 1, 2: 2}],
                block_size=4,
                seq_lens=[10],
                out_cache_loc=[8],
                sliding_window=6,
            )

    def test_req_to_token_table_rejects_bad_metadata(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import ReqToTokenTable

        table = ReqToTokenTable(max_requests=1, max_context_len=3)
        slot = table.allocate("a")
        table.append_slots(slot, [4, 5])
        with self.assertRaisesRegex(ValueError, "exceed"):
            table.build_decode_metadata([slot], seq_lens=[3])
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            table.append_slots(slot, [6, 7])
        with self.assertRaisesRegex(KeyError, "not allocated"):
            table.build_decode_metadata([table.padding_req_slot])

    def test_req_to_token_table_rejects_padding_in_paged_metadata(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import ReqToTokenTable

        table = ReqToTokenTable(max_requests=1, max_context_len=4)
        slot = table.allocate("a")
        table.append_slots(slot, [0, 1, 2])
        table.clear_before(slot, 1)

        with self.assertRaisesRegex(RuntimeError, "padding"):
            table.build_paged_decode_metadata([slot], block_size=4, out_cache_loc=[2])

    def test_req_to_token_table_rejects_bad_paged_metadata_layouts(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import ReqToTokenTable

        duplicate = ReqToTokenTable(max_requests=1, max_context_len=2)
        duplicate_slot = duplicate.allocate("dup")
        duplicate.append_slots(duplicate_slot, [0, 0])
        with self.assertRaisesRegex(ValueError, "duplicate slots"):
            duplicate.build_paged_decode_metadata(
                [duplicate_slot],
                block_size=4,
                out_cache_loc=[0],
            )

        unaligned = ReqToTokenTable(max_requests=1, max_context_len=2)
        unaligned_slot = unaligned.allocate("unaligned")
        unaligned.append_slots(unaligned_slot, [8, 10])
        with self.assertRaisesRegex(ValueError, "page-aligned"):
            unaligned.build_paged_decode_metadata(
                [unaligned_slot],
                block_size=4,
                out_cache_loc=[10],
            )

        non_final_out = ReqToTokenTable(max_requests=1, max_context_len=3)
        non_final_slot = non_final_out.allocate("non-final")
        non_final_out.append_slots(non_final_slot, [0, 1, 2])
        with self.assertRaisesRegex(ValueError, "final logical token"):
            non_final_out.build_paged_decode_metadata(
                [non_final_slot],
                block_size=4,
                out_cache_loc=[1],
            )

    def test_explicit_token_slot_rows_preserve_nonchronological_order(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            build_decode_metadata_from_token_slot_rows,
        )

        metadata = build_decode_metadata_from_token_slot_rows(
            [
                torch.tensor([30, 10, 11, 40], dtype=torch.int64),
                [7, 5],
            ],
            req_slots=[3, 4],
            logical_seq_lens=[128, 96],
            out_cache_loc=[40, 5],
            token_pool_capacity=64,
        )

        self.assertEqual(metadata.req_pool_indices.tolist(), [3, 4])
        self.assertEqual(metadata.seq_lens.tolist(), [4, 2])
        self.assertEqual(metadata.logical_seq_lens.tolist(), [128, 96])
        self.assertEqual(metadata.out_cache_loc.tolist(), [40, 5])
        self.assertEqual(metadata.out_cache_loc_long.tolist(), [40, 5])
        self.assertEqual(metadata.out_cache_loc_long.dtype, torch.long)
        self.assertEqual(metadata.kv_indptr.tolist(), [0, 4, 6])
        self.assertEqual(metadata.kv_indices.tolist(), [30, 10, 11, 40, 7, 5])
        self.assertEqual(metadata.kv_indices.dtype, torch.int32)

    def test_explicit_token_slot_rows_reuses_decode_metadata_workspace(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            build_decode_metadata_from_token_slot_rows,
        )

        workspace = {}
        first = build_decode_metadata_from_token_slot_rows(
            [
                torch.tensor([30, 10, 11, 40], dtype=torch.int64),
                [7, 5],
            ],
            req_slots=[3, 4],
            logical_seq_lens=[128, 96],
            out_cache_loc=[40, 5],
            token_pool_capacity=64,
            workspace=workspace,
            kv_indices_padding_slots=2,
        )
        ptrs = {
            name: int(getattr(first, name).data_ptr())
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

        second = build_decode_metadata_from_token_slot_rows(
            [
                torch.tensor([31, 12, 13, 41], dtype=torch.int64),
                [8, 6],
            ],
            req_slots=[5, 6],
            logical_seq_lens=[129, 97],
            out_cache_loc=[41, 6],
            token_pool_capacity=64,
            workspace=workspace,
            kv_indices_padding_slots=2,
        )

        for name, ptr in ptrs.items():
            self.assertEqual(int(getattr(second, name).data_ptr()), ptr)
        self.assertEqual(second.req_pool_indices.tolist(), [5, 6])
        self.assertEqual(second.seq_lens.tolist(), [4, 2])
        self.assertEqual(second.logical_seq_lens.tolist(), [129, 97])
        self.assertEqual(second.out_cache_loc.tolist(), [41, 6])
        self.assertEqual(second.out_cache_loc_long.tolist(), [41, 6])
        self.assertEqual(second.out_cache_loc_long.dtype, torch.long)
        self.assertEqual(second.kv_indptr.tolist(), [0, 4, 6])
        self.assertEqual(second.kv_indices.tolist(), [31, 12, 13, 41, 8, 6, 6, 6])

    def test_explicit_token_slot_rows_accept_typed_decode_metadata_workspace(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            TokenPoolDecodeMetadataWorkspace,
            build_decode_metadata_from_token_slot_rows,
        )

        workspace = TokenPoolDecodeMetadataWorkspace()
        first = build_decode_metadata_from_token_slot_rows(
            [[10, 11], [20, 21]],
            req_slots=[1, 2],
            logical_seq_lens=[8, 9],
            out_cache_loc=[11, 21],
            workspace=workspace,
            workspace_key="full",
        )
        first_ptr = int(first.kv_indices.data_ptr())
        second = build_decode_metadata_from_token_slot_rows(
            [[12, 13], [22, 23]],
            req_slots=[3, 4],
            logical_seq_lens=[10, 11],
            out_cache_loc=[13, 23],
            workspace=workspace,
            workspace_key="full",
        )
        self.assertEqual(int(second.kv_indices.data_ptr()), first_ptr)
        self.assertEqual(second.kv_indices.tolist(), [12, 13, 22, 23])
        self.assertEqual(second.req_pool_indices.tolist(), [3, 4])
        self.assertEqual(second.out_cache_loc_long.dtype, torch.long)

        sliding = build_decode_metadata_from_token_slot_rows(
            [[30, 31]],
            req_slots=[5],
            logical_seq_lens=[12],
            out_cache_loc=[31],
            workspace=workspace,
            workspace_key="sliding",
        )
        self.assertNotEqual(int(sliding.kv_indices.data_ptr()), first_ptr)
        self.assertIs(
            workspace.flat_workspaces["full"],
            workspace.flat_workspace("full"),
        )
        self.assertIn("sliding", workspace.flat_workspaces)

    def test_explicit_token_slot_row_chunks_fill_workspace_without_cat(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            TokenSlotRowChunks,
            build_decode_metadata_from_token_slot_rows,
        )

        workspace = {}
        original_cat = torch.cat

        def fail_cat(*_args, **_kwargs):
            raise AssertionError("trusted workspace row chunks should not call torch.cat")

        torch.cat = fail_cat
        try:
            metadata = build_decode_metadata_from_token_slot_rows(
                [
                    TokenSlotRowChunks(
                        (
                            torch.tensor([30, 10, 11], dtype=torch.int32),
                            torch.tensor([40], dtype=torch.int32),
                        ),
                        trusted=True,
                    ),
                    torch.tensor([7, 5], dtype=torch.int32),
                ],
                req_slots=[3, 4],
                logical_seq_lens=[128, 96],
                out_cache_loc=[40, 5],
                token_pool_capacity=64,
                workspace=workspace,
                kv_indices_padding_slots=1,
            )
        finally:
            torch.cat = original_cat

        self.assertEqual(metadata.req_pool_indices.tolist(), [3, 4])
        self.assertEqual(metadata.seq_lens.tolist(), [4, 2])
        self.assertEqual(metadata.logical_seq_lens.tolist(), [128, 96])
        self.assertEqual(metadata.out_cache_loc.tolist(), [40, 5])
        self.assertEqual(metadata.kv_indptr.tolist(), [0, 4, 6])
        self.assertEqual(metadata.kv_indices.tolist(), [30, 10, 11, 40, 7, 5, 5])

    def test_trusted_explicit_rows_skip_aux_metadata_uniqueness_checks(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            TokenSlotRowChunks,
            build_decode_metadata_from_token_slot_rows,
        )

        workspace = {}
        original_unique = torch.unique

        def fail_unique(*_args, **_kwargs):
            raise AssertionError("trusted engine metadata should not call torch.unique")

        torch.unique = fail_unique
        try:
            metadata = build_decode_metadata_from_token_slot_rows(
                [
                    TokenSlotRowChunks(
                        (torch.tensor([30, 10, 40], dtype=torch.int32),),
                        trusted=True,
                    ),
                    TokenSlotRowChunks(
                        (
                            torch.tensor([7], dtype=torch.int32),
                            torch.tensor([5], dtype=torch.int32),
                        ),
                        trusted=True,
                    ),
                ],
                req_slots=[3, 4],
                logical_seq_lens=[128, 96],
                out_cache_loc=[40, 5],
                token_pool_capacity=64,
                workspace=workspace,
                trusted_aux_metadata=True,
            )
        finally:
            torch.unique = original_unique

        self.assertEqual(metadata.req_pool_indices.tolist(), [3, 4])
        self.assertEqual(metadata.logical_seq_lens.tolist(), [128, 96])
        self.assertEqual(metadata.out_cache_loc.tolist(), [40, 5])
        self.assertEqual(metadata.kv_indices.tolist(), [30, 10, 40, 7, 5])

    def test_explicit_token_slot_row_chunks_reject_duplicate_slots_by_default(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            TokenSlotRowChunks,
            build_decode_metadata_from_token_slot_rows,
        )

        with self.assertRaisesRegex(ValueError, "duplicate slots"):
            build_decode_metadata_from_token_slot_rows(
                [
                    TokenSlotRowChunks(
                        (
                            torch.tensor([1, 2], dtype=torch.int32),
                            torch.tensor([2, 3], dtype=torch.int32),
                        ),
                    )
                ],
            )

    def test_explicit_token_slot_rows_default_logical_lengths_and_req_slots(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            build_decode_metadata_from_slot_sequences,
        )

        metadata = build_decode_metadata_from_slot_sequences(
            [[2, 1], torch.tensor([4, 3, 0], dtype=torch.int32)],
        )

        self.assertEqual(metadata.req_pool_indices.tolist(), [0, 1])
        self.assertEqual(metadata.seq_lens.tolist(), [2, 3])
        self.assertEqual(metadata.logical_seq_lens.tolist(), [2, 3])
        self.assertIsNone(metadata.out_cache_loc)
        self.assertEqual(metadata.kv_indptr.tolist(), [0, 2, 5])
        self.assertEqual(metadata.kv_indices.tolist(), [2, 1, 4, 3, 0])

    def test_explicit_token_slot_rows_reject_ambiguous_layouts(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            build_decode_metadata_from_token_slot_rows,
        )

        with self.assertRaisesRegex(ValueError, "at least one"):
            build_decode_metadata_from_token_slot_rows([])
        with self.assertRaisesRegex(ValueError, "non-empty"):
            build_decode_metadata_from_token_slot_rows([[]])
        with self.assertRaisesRegex(ValueError, "negative"):
            build_decode_metadata_from_token_slot_rows([[1, -2]])
        with self.assertRaisesRegex(ValueError, "duplicate slots"):
            build_decode_metadata_from_token_slot_rows([[1, 1]])
        with self.assertRaisesRegex(ValueError, "req_slots length"):
            build_decode_metadata_from_token_slot_rows([[1]], req_slots=[0, 1])
        with self.assertRaisesRegex(ValueError, "req_slots must be unique"):
            build_decode_metadata_from_token_slot_rows([[1], [2]], req_slots=[0, 0])
        with self.assertRaisesRegex(ValueError, "logical_seq_lens length"):
            build_decode_metadata_from_token_slot_rows([[1]], logical_seq_lens=[1, 2])
        with self.assertRaisesRegex(ValueError, "out_cache_loc length"):
            build_decode_metadata_from_token_slot_rows([[1]], out_cache_loc=[1, 2])
        with self.assertRaisesRegex(ValueError, "out_cache_loc must be unique"):
            build_decode_metadata_from_token_slot_rows(
                [[1, 2], [3, 4]],
                out_cache_loc=[2, 2],
            )
        with self.assertRaisesRegex(ValueError, "exactly once"):
            build_decode_metadata_from_token_slot_rows([[1, 2]], out_cache_loc=[3])
        with self.assertRaisesRegex(ValueError, "token_pool_capacity"):
            build_decode_metadata_from_token_slot_rows(
                [[1, 4]],
                out_cache_loc=[4],
                token_pool_capacity=4,
            )

    def test_paged_token_slot_rows_build_selected_window_block_tables(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            build_paged_decode_metadata_from_token_slot_rows,
        )

        metadata = build_paged_decode_metadata_from_token_slot_rows(
            [
                [0, 1, 2, 3, 8, 9],
                torch.tensor([16, 17, 18, 19], dtype=torch.int64),
            ],
            block_size=4,
            req_slots=[3, 4],
            logical_seq_lens=[6, 4],
            out_cache_loc=[9, 19],
            token_pool_capacity=24,
            padding_block=-7,
        )

        self.assertEqual(metadata.req_pool_indices.tolist(), [3, 4])
        self.assertEqual(metadata.seq_lens.tolist(), [6, 4])
        self.assertEqual(metadata.logical_seq_lens.tolist(), [6, 4])
        self.assertEqual(metadata.out_cache_loc.tolist(), [9, 19])
        self.assertEqual(metadata.out_cache_loc_long.tolist(), [9, 19])
        self.assertEqual(metadata.out_cache_loc_long.dtype, torch.long)
        self.assertEqual(metadata.slot_mapping.tolist(), [9, 19])
        self.assertEqual(metadata.slot_mapping.dtype, torch.long)
        self.assertEqual(metadata.selected_start_positions.tolist(), [0, 0])
        self.assertEqual(metadata.block_size, 4)
        self.assertEqual(metadata.block_table_lens.tolist(), [2, 1])
        self.assertEqual(metadata.block_tables.tolist(), [[0, 2], [4, -7]])

    def test_paged_token_slot_rows_support_default_and_mid_block_windows(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            build_paged_decode_metadata_from_token_slot_rows,
        )

        default_start = build_paged_decode_metadata_from_token_slot_rows(
            [[8, 9, 10, 11, 16, 17]],
            block_size=4,
            logical_seq_lens=[14],
            out_cache_loc=[17],
        )
        self.assertEqual(default_start.seq_lens.tolist(), [6])
        self.assertEqual(default_start.logical_seq_lens.tolist(), [14])
        self.assertEqual(default_start.selected_start_positions.tolist(), [8])
        self.assertEqual(default_start.block_table_lens.tolist(), [2])
        self.assertEqual(default_start.block_tables.tolist(), [[2, 4]])

        padded = build_paged_decode_metadata_from_token_slot_rows(
            [[8, 9, 10, 11, 16, 17]],
            block_size=4,
            block_table_width=3,
            logical_seq_lens=[14],
            out_cache_loc=[17],
            padding_block=-5,
        )
        self.assertEqual(padded.block_table_lens.tolist(), [2])
        self.assertEqual(padded.block_tables.tolist(), [[2, 4, -5]])

        mid_block = build_paged_decode_metadata_from_token_slot_rows(
            [[6, 7, 8, 9]],
            block_size=4,
            logical_seq_lens=[10],
            selected_start_positions=[6],
            out_cache_loc=[9],
        )
        self.assertEqual(mid_block.seq_lens.tolist(), [4])
        self.assertEqual(mid_block.logical_seq_lens.tolist(), [10])
        self.assertEqual(mid_block.selected_start_positions.tolist(), [6])
        self.assertEqual(mid_block.block_table_lens.tolist(), [2])
        self.assertEqual(mid_block.block_tables.tolist(), [[1, 2]])

    def test_paged_token_slot_rows_reject_bad_block_metadata(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            build_paged_decode_metadata_from_token_slot_rows,
        )

        with self.assertRaisesRegex(ValueError, "at least one"):
            build_paged_decode_metadata_from_token_slot_rows([], block_size=4)
        with self.assertRaisesRegex(ValueError, "block_size"):
            build_paged_decode_metadata_from_token_slot_rows([[0]], block_size=0)
        with self.assertRaisesRegex(ValueError, "non-empty"):
            build_paged_decode_metadata_from_token_slot_rows([[]], block_size=4)
        with self.assertRaisesRegex(ValueError, "negative"):
            build_paged_decode_metadata_from_token_slot_rows([[0, -1]], block_size=4)
        with self.assertRaisesRegex(ValueError, "duplicate slots"):
            build_paged_decode_metadata_from_token_slot_rows([[0, 0]], block_size=4)
        with self.assertRaisesRegex(ValueError, "req_slots length"):
            build_paged_decode_metadata_from_token_slot_rows(
                [[0]],
                block_size=4,
                req_slots=[0, 1],
            )
        with self.assertRaisesRegex(ValueError, "logical_seq_lens length"):
            build_paged_decode_metadata_from_token_slot_rows(
                [[0]],
                block_size=4,
                logical_seq_lens=[1, 2],
            )
        with self.assertRaisesRegex(ValueError, "selected_start_positions length"):
            build_paged_decode_metadata_from_token_slot_rows(
                [[0]],
                block_size=4,
                selected_start_positions=[0, 1],
            )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            build_paged_decode_metadata_from_token_slot_rows(
                [[0, 1]],
                block_size=4,
                logical_seq_lens=[1],
            )
        with self.assertRaisesRegex(ValueError, "exceeds logical length"):
            build_paged_decode_metadata_from_token_slot_rows(
                [[0, 1]],
                block_size=4,
                logical_seq_lens=[1],
                selected_start_positions=[0],
            )
        with self.assertRaisesRegex(ValueError, "page-aligned"):
            build_paged_decode_metadata_from_token_slot_rows(
                [[8, 10]],
                block_size=4,
                logical_seq_lens=[6],
            )
        with self.assertRaisesRegex(ValueError, "multiple physical blocks"):
            build_paged_decode_metadata_from_token_slot_rows(
                [[0, 5]],
                block_size=4,
                logical_seq_lens=[2],
            )
        with self.assertRaisesRegex(ValueError, "out_cache_loc must be unique"):
            build_paged_decode_metadata_from_token_slot_rows(
                [[0], [4]],
                block_size=4,
                out_cache_loc=[0, 0],
            )
        with self.assertRaisesRegex(ValueError, "exactly once"):
            build_paged_decode_metadata_from_token_slot_rows(
                [[0, 1]],
                block_size=4,
                out_cache_loc=[2],
            )
        with self.assertRaisesRegex(ValueError, "inside the selected decode row"):
            build_paged_decode_metadata_from_token_slot_rows(
                [[6, 7]],
                block_size=4,
                logical_seq_lens=[10],
                selected_start_positions=[6],
                out_cache_loc=[7],
            )
        with self.assertRaisesRegex(ValueError, "final logical token"):
            build_paged_decode_metadata_from_token_slot_rows(
                [[100, 101, 102]],
                block_size=4,
                out_cache_loc=[100],
            )
        with self.assertRaisesRegex(ValueError, "block_table_width"):
            build_paged_decode_metadata_from_token_slot_rows(
                [[0, 1, 2, 3, 4]],
                block_size=4,
                block_table_width=1,
                out_cache_loc=[4],
            )
        with self.assertRaisesRegex(ValueError, "token_pool_capacity"):
            build_paged_decode_metadata_from_token_slot_rows(
                [[0, 1]],
                block_size=4,
                out_cache_loc=[1],
                token_pool_capacity=1,
            )

    def test_paged_token_slot_rows_can_allow_materialized_rows_past_logical_len(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            build_paged_decode_metadata_from_token_slot_rows,
        )

        with self.assertRaisesRegex(ValueError, "exceeds logical length"):
            build_paged_decode_metadata_from_token_slot_rows(
                [[0, 1, 2, 3]],
                block_size=2,
                logical_seq_lens=[2],
                selected_start_positions=[0],
            )

        metadata = build_paged_decode_metadata_from_token_slot_rows(
            [[0, 1, 2, 3]],
            block_size=2,
            logical_seq_lens=[2],
            out_cache_loc=[3],
            selected_start_positions=[0],
            allow_selected_len_gt_logical_len=True,
            max_seq_len=6,
        )

        self.assertEqual(metadata.seq_lens.tolist(), [4])
        self.assertEqual(metadata.logical_seq_lens.tolist(), [2])
        self.assertEqual(metadata.block_tables.tolist(), [[0, 1]])
        self.assertEqual(metadata.block_table_lens.tolist(), [2])
        self.assertEqual(metadata.slot_mapping.tolist(), [3])
        self.assertEqual(metadata.max_seq_len, 6)

        with self.assertRaisesRegex(ValueError, "final logical token"):
            build_paged_decode_metadata_from_token_slot_rows(
                [[0, 1, 2, 3]],
                block_size=2,
                logical_seq_lens=[2],
                out_cache_loc=[1],
                selected_start_positions=[0],
                allow_selected_len_gt_logical_len=True,
            )
        with self.assertRaisesRegex(ValueError, "max_seq_len"):
            build_paged_decode_metadata_from_token_slot_rows(
                [[0, 1, 2, 3]],
                block_size=2,
                logical_seq_lens=[2],
                selected_start_positions=[0],
                allow_selected_len_gt_logical_len=True,
                max_seq_len=3,
            )

    def test_paged_triton_decode_matches_manual_grouped_attention(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")

        from wkvm.runner.gemma_token_pool_triton import token_pool_paged_gqa_decode

        torch.manual_seed(123)
        device = torch.device("cuda")
        batch = 2
        query_heads = 4
        kv_heads = 2
        groups = query_heads // kv_heads
        head_dim = 16
        block_size = 4
        scaling = head_dim ** -0.5

        query = torch.randn(
            batch,
            query_heads,
            1,
            head_dim,
            dtype=torch.float32,
            device=device,
        )
        keys = torch.randn(
            24,
            kv_heads,
            head_dim,
            dtype=torch.float32,
            device=device,
        )
        values = torch.randn_like(keys)
        block_tables = torch.tensor(
            [[0, 2], [1, 2]],
            dtype=torch.int32,
            device=device,
        )
        block_table_lens = torch.tensor([2, 2], dtype=torch.int32, device=device)
        selected_start_positions = torch.tensor([0, 6], dtype=torch.int32, device=device)
        seq_lens = torch.tensor([6, 4], dtype=torch.int32, device=device)

        actual = token_pool_paged_gqa_decode(
            query,
            keys,
            values,
            block_tables,
            block_table_lens,
            selected_start_positions,
            seq_lens,
            block_size=block_size,
            num_key_value_groups=groups,
            scaling=scaling,
            block_n=16,
        )
        torch.cuda.synchronize()

        expected_rows = []
        for row, token_indices in enumerate(([0, 1, 2, 3, 8, 9], [6, 7, 8, 9])):
            row_outputs = []
            row_keys = keys[torch.tensor(token_indices, dtype=torch.long, device=device)]
            row_values = values[torch.tensor(token_indices, dtype=torch.long, device=device)]
            for q_head in range(query_heads):
                kv_head = q_head // groups
                scores = (
                    row_keys[:, kv_head, :] * query[row, q_head, 0, :].unsqueeze(0)
                ).sum(dim=-1) * scaling
                probs = torch.softmax(scores, dim=-1, dtype=torch.float32)
                row_outputs.append((probs.unsqueeze(0) @ row_values[:, kv_head, :]).squeeze(0))
            expected_rows.append(torch.stack(row_outputs, dim=0))
        expected = torch.stack(expected_rows, dim=0).unsqueeze(1)

        self.assertLess((expected - actual).abs().max().item(), 1e-5)

    def test_split_kv_triton_decode_matches_flat_grouped_kernel(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")

        from wkvm.runner.gemma_token_pool_triton import (
            token_pool_gqa_decode,
            token_pool_gqa_decode_split_kv,
        )

        torch.manual_seed(812)
        device = torch.device("cuda")
        batch = 3
        query_heads = 4
        kv_heads = 2
        groups = query_heads // kv_heads
        head_dim = 16
        scaling = head_dim ** -0.5

        query = torch.randn(
            batch,
            query_heads,
            1,
            head_dim,
            dtype=torch.float32,
            device=device,
        )
        keys = torch.randn(
            64,
            kv_heads,
            head_dim,
            dtype=torch.float32,
            device=device,
        )
        values = torch.randn_like(keys)
        kv_indices = torch.tensor(
            [0, 2, 4, 6, 8, 10, 12, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19],
            dtype=torch.int32,
            device=device,
        )
        kv_indptr = torch.tensor([0, 7, 12, 17], dtype=torch.int32, device=device)

        expected = token_pool_gqa_decode(
            query,
            keys,
            values,
            kv_indptr,
            kv_indices,
            num_key_value_groups=groups,
            scaling=scaling,
            block_n=16,
        )
        workspace = (
            torch.full((batch, kv_heads, 4, groups), float("nan"), device=device),
            torch.full((batch, kv_heads, 4, groups), float("nan"), device=device),
            torch.full((batch, kv_heads, 4, groups, head_dim), float("nan"), device=device),
        )
        actual = token_pool_gqa_decode_split_kv(
            query,
            keys,
            values,
            kv_indptr,
            kv_indices,
            num_key_value_groups=groups,
            scaling=scaling,
            max_seq_len=12,
            split_size=3,
            min_splits=2,
            block_n=16,
            seq_lens=kv_indptr[1:] - kv_indptr[:-1],
            workspace=workspace,
        )
        torch.cuda.synchronize()

        self.assertLess((expected - actual).abs().max().item(), 1e-5)
        with self.assertRaisesRegex(ValueError, "seq_lens length"):
            token_pool_gqa_decode_split_kv(
                query,
                keys,
                values,
                kv_indptr,
                kv_indices,
                num_key_value_groups=groups,
                scaling=scaling,
                max_seq_len=12,
                split_size=3,
                min_splits=2,
                block_n=16,
                seq_lens=torch.tensor([7, 5], dtype=torch.int32, device=device),
            )

    def test_paged_split_kv_triton_decode_matches_paged_grouped_kernel(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")

        from wkvm.runner.gemma_token_pool_triton import (
            token_pool_paged_gqa_decode,
            token_pool_paged_gqa_decode_split_kv,
        )

        torch.manual_seed(913)
        device = torch.device("cuda")
        batch = 2
        query_heads = 4
        kv_heads = 2
        groups = query_heads // kv_heads
        head_dim = 16
        block_size = 4
        scaling = head_dim ** -0.5

        query = torch.randn(
            batch,
            query_heads,
            1,
            head_dim,
            dtype=torch.float32,
            device=device,
        )
        keys = torch.randn(
            24,
            kv_heads,
            head_dim,
            dtype=torch.float32,
            device=device,
        )
        values = torch.randn_like(keys)
        block_tables = torch.tensor(
            [[0, 2], [1, 2]],
            dtype=torch.int32,
            device=device,
        )
        block_table_lens = torch.tensor([2, 2], dtype=torch.int32, device=device)
        selected_start_positions = torch.tensor([0, 6], dtype=torch.int32, device=device)
        seq_lens = torch.tensor([6, 4], dtype=torch.int32, device=device)

        expected = token_pool_paged_gqa_decode(
            query,
            keys,
            values,
            block_tables,
            block_table_lens,
            selected_start_positions,
            seq_lens,
            block_size=block_size,
            num_key_value_groups=groups,
            scaling=scaling,
            block_n=16,
        )
        workspace = (
            torch.full((batch, kv_heads, 3, groups), float("nan"), device=device),
            torch.full((batch, kv_heads, 3, groups), float("nan"), device=device),
            torch.full((batch, kv_heads, 3, groups, head_dim), float("nan"), device=device),
        )
        actual = token_pool_paged_gqa_decode_split_kv(
            query,
            keys,
            values,
            block_tables,
            block_table_lens,
            selected_start_positions,
            seq_lens,
            block_size=block_size,
            num_key_value_groups=groups,
            scaling=scaling,
            max_seq_len=8,
            split_size=3,
            min_splits=2,
            block_n=16,
            workspace=workspace,
        )
        torch.cuda.synchronize()

        self.assertLess((expected - actual).abs().max().item(), 1e-5)

    def test_token_pool_decode_context_prefers_layer_id_metadata(self) -> None:
        from types import SimpleNamespace

        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            build_decode_metadata_from_token_slot_rows,
            build_paged_decode_metadata_from_token_slot_rows,
            TokenPoolDecodeContext,
        )

        by_type = build_decode_metadata_from_token_slot_rows(
            [[1, 2]],
            out_cache_loc=[2],
        )
        by_layer = build_decode_metadata_from_token_slot_rows(
            [[3, 4]],
            out_cache_loc=[4],
        )
        paged_by_type = build_paged_decode_metadata_from_token_slot_rows(
            [[0, 1]],
            block_size=4,
            out_cache_loc=[1],
        )
        paged_by_layer = build_paged_decode_metadata_from_token_slot_rows(
            [[4, 5]],
            block_size=4,
            logical_seq_lens=[6],
            selected_start_positions=[4],
            out_cache_loc=[5],
        )
        context = TokenPoolDecodeContext(
            metadata_by_layer_type={"full_attention": by_type},
            metadata_by_layer_id={7: by_layer},
            paged_metadata_by_layer_type={"full_attention": paged_by_type},
            paged_metadata_by_layer_id={7: paged_by_layer},
        )

        self.assertIs(context.metadata_for_layer(7, "full_attention"), by_layer)
        self.assertIs(context.metadata_for_layer(8, "full_attention"), by_type)
        self.assertIsNone(context.metadata_for_layer(8, None))
        self.assertIs(
            context.paged_metadata_for_layer(7, "full_attention"),
            paged_by_layer,
        )
        self.assertIs(
            context.paged_metadata_for_layer(8, "full_attention"),
            paged_by_type,
        )
        self.assertIsNone(context.paged_metadata_for_layer(8, None))
        resolved_flat, resolved_paged, resolved_pool = context.attention_metadata_for_layer(
            7,
            "full_attention",
        )
        self.assertIs(resolved_flat, by_layer)
        self.assertIs(resolved_paged, paged_by_layer)
        self.assertIsNone(resolved_pool)
        resolved_flat, resolved_paged, resolved_pool = context.attention_metadata_for_layer(
            8,
            "full_attention",
        )
        self.assertIs(resolved_flat, by_type)
        self.assertIs(resolved_paged, paged_by_type)
        self.assertIsNone(resolved_pool)

        pool = SimpleNamespace(layer_specs={7: object()})
        context = TokenPoolDecodeContext(
            metadata_by_layer_type={"full_attention": by_type},
            metadata_by_layer_id={7: by_layer},
            paged_metadata_by_layer_type={"full_attention": paged_by_type},
            paged_metadata_by_layer_id={7: paged_by_layer},
            kv_pool=pool,
        )
        resolved_flat, resolved_paged, resolved_pool = context.attention_metadata_for_layer(
            7,
            "full_attention",
        )
        self.assertIs(resolved_flat, by_layer)
        self.assertIs(resolved_paged, paged_by_layer)
        self.assertIs(resolved_pool, pool)
        with self.assertRaisesRegex(RuntimeError, "KV pool has no spec"):
            context.attention_metadata_for_layer(8, "full_attention")
        resolved_flat, resolved_paged, resolved_pool = context.attention_metadata_for_layer(
            8,
            "full_attention",
            attention_mask_present=True,
        )
        self.assertIsNone(resolved_flat)
        self.assertIsNone(resolved_paged)
        self.assertIsNone(resolved_pool)

    def test_attention_binding_owns_current_kv_write(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            DecodeBatchMetadata,
            TokenPoolAttentionBinding,
            TokenPoolDecodeContext,
            build_decode_metadata_from_token_slot_rows,
        )

        class CapturePool:
            layer_specs = {7: object()}

            def __init__(self) -> None:
                self.calls = []

            def set_kv(self, layer_idx, out_cache_loc, key_states, value_states):
                self.calls.append(
                    (layer_idx, out_cache_loc, key_states, value_states)
                )

        metadata = build_decode_metadata_from_token_slot_rows(
            [[3, 4]],
            out_cache_loc=[4],
        )
        pool = CapturePool()
        context = TokenPoolDecodeContext(
            metadata_by_layer_type={"full_attention": metadata},
            kv_pool=pool,
        )
        binding = context.attention_binding_for_layer(7, "full_attention")

        key_states = object()
        value_states = object()
        out_cache_loc = binding.store_current_kv(key_states, value_states)

        self.assertIs(binding.metadata, metadata)
        self.assertIs(binding.kv_pool, pool)
        self.assertIs(out_cache_loc, metadata.out_cache_loc_long)
        self.assertEqual(len(pool.calls), 1)
        layer_idx, written_slots, written_keys, written_values = pool.calls[0]
        self.assertEqual(layer_idx, 7)
        self.assertIs(written_slots, metadata.out_cache_loc_long)
        self.assertIs(written_keys, key_states)
        self.assertIs(written_values, value_states)

        fallback_metadata = DecodeBatchMetadata(
            req_pool_indices=metadata.req_pool_indices,
            seq_lens=metadata.seq_lens,
            logical_seq_lens=metadata.logical_seq_lens,
            out_cache_loc=metadata.out_cache_loc,
            kv_indptr=metadata.kv_indptr,
            kv_indices=metadata.kv_indices,
            out_cache_loc_long=None,
            max_seq_len=metadata.max_seq_len,
        )
        fallback_binding = TokenPoolAttentionBinding(
            layer_idx=7,
            metadata=fallback_metadata,
            paged_metadata=None,
            kv_pool=pool,
        )
        self.assertIs(
            fallback_binding.out_cache_loc_for_write(),
            fallback_metadata.out_cache_loc,
        )
        null_binding = TokenPoolAttentionBinding(
            layer_idx=None,
            metadata=None,
            paged_metadata=None,
            kv_pool=None,
        )
        self.assertIsNone(null_binding.store_current_kv(key_states, value_states))

    def test_attention_plan_resolves_decode_eligibility_and_write(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            DecodeBatchMetadata,
            TokenPoolAttentionBinding,
            TokenPoolAttentionPlan,
            TokenPoolDecodeContext,
            build_decode_metadata_from_token_slot_rows,
            resolve_token_pool_attention_plan,
        )

        class CapturePool:
            layer_specs = {7: object()}

            def __init__(self) -> None:
                self.calls = []

            def set_kv(self, layer_idx, out_cache_loc, key_states, value_states):
                self.calls.append(
                    (layer_idx, out_cache_loc, key_states, value_states)
                )

        metadata = build_decode_metadata_from_token_slot_rows(
            [[3, 4]],
            out_cache_loc=[4],
        )
        pool = CapturePool()
        context = TokenPoolDecodeContext(
            metadata_by_layer_type={"full_attention": metadata},
            kv_pool=pool,
        )

        plan = context.attention_plan_for_layer(
            7,
            "full_attention",
            query_seq_len=1,
        )

        self.assertTrue(plan.use_decode_attention)
        self.assertIs(plan.metadata, metadata)
        self.assertIs(plan.kv_pool, pool)
        self.assertEqual(
            plan.attention_kwargs(),
            {
                "decode_metadata": metadata,
                "paged_decode_metadata": None,
                "token_kv_pool": pool,
                "layer_idx": 7,
            },
        )

        key_states = object()
        value_states = object()
        out_cache_loc = plan.store_current_kv(key_states, value_states)

        self.assertIs(out_cache_loc, metadata.out_cache_loc_long)
        self.assertEqual(len(pool.calls), 1)
        self.assertEqual(pool.calls[0][0], 7)
        self.assertIs(pool.calls[0][1], metadata.out_cache_loc_long)
        self.assertIs(pool.calls[0][2], key_states)
        self.assertIs(pool.calls[0][3], value_states)

        masked_plan = context.attention_plan_for_layer(
            7,
            "full_attention",
            attention_mask_present=True,
            query_seq_len=1,
        )
        self.assertFalse(masked_plan.use_decode_attention)
        multi_token_plan = context.attention_plan_for_layer(
            7,
            "full_attention",
            query_seq_len=2,
        )
        self.assertFalse(multi_token_plan.use_decode_attention)
        no_write_metadata = DecodeBatchMetadata(
            req_pool_indices=metadata.req_pool_indices,
            seq_lens=metadata.seq_lens,
            logical_seq_lens=metadata.logical_seq_lens,
            out_cache_loc=None,
            kv_indptr=metadata.kv_indptr,
            kv_indices=metadata.kv_indices,
        )
        no_write_plan = TokenPoolAttentionPlan.from_binding(
            TokenPoolAttentionBinding(
                layer_idx=7,
                metadata=no_write_metadata,
                paged_metadata=None,
                kv_pool=pool,
            ),
            layer_idx=7,
            query_seq_len=1,
        )
        self.assertFalse(no_write_plan.use_decode_attention)
        self.assertIsNone(no_write_plan.store_current_kv(key_states, value_states))

        legacy_context = type(
            "LegacyContext",
            (),
            {
                "kv_pool": pool,
                "metadata_by_layer_type": {"full_attention": metadata},
            },
        )()
        legacy_plan = resolve_token_pool_attention_plan(
            legacy_context,
            7,
            "full_attention",
            query_seq_len=1,
        )
        self.assertTrue(legacy_plan.use_decode_attention)
        self.assertIs(legacy_plan.metadata, metadata)
        self.assertIs(legacy_plan.kv_pool, pool)

        class BindingWithOwnStore:
            def __init__(self) -> None:
                self.layer_idx = 7
                self.metadata = metadata
                self.paged_metadata = None
                self.kv_pool = pool
                self.calls = []

            def out_cache_loc_for_write(self):
                return self.metadata.out_cache_loc

            def store_current_kv(self, key_states, value_states):
                self.calls.append((key_states, value_states))
                return self.metadata.out_cache_loc

        owned_binding = BindingWithOwnStore()
        owned_plan = TokenPoolAttentionPlan.from_binding(
            owned_binding,
            layer_idx=7,
            query_seq_len=1,
        )
        previous_pool_calls = len(pool.calls)
        self.assertIs(
            owned_plan.store_current_kv(key_states, value_states),
            metadata.out_cache_loc,
        )
        self.assertEqual(owned_binding.calls, [(key_states, value_states)])
        self.assertEqual(len(pool.calls), previous_pool_calls)

    def test_graph_clone_and_copy_handles_paged_metadata(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_runner import _GraphedPaddedDecodeStep
        from wkvm.runner.gemma_token_pool import (
            build_decode_metadata_from_token_slot_rows,
            build_paged_decode_metadata_from_token_slot_rows,
            TokenPoolDecodeContext,
        )

        flat = build_decode_metadata_from_token_slot_rows([[0, 1]], out_cache_loc=[1])
        paged = build_paged_decode_metadata_from_token_slot_rows(
            [[0, 1]],
            block_size=4,
            out_cache_loc=[1],
        )
        context = TokenPoolDecodeContext(
            metadata_by_layer_type={"sliding_attention": flat},
            metadata_by_layer_id={0: flat, 1: flat},
            paged_metadata_by_layer_type={"sliding_attention": paged},
            paged_metadata_by_layer_id={0: paged, 1: paged},
            kv_pool=object(),
            covered_layer_types=frozenset({"sliding_attention"}),
        )

        cloned = _GraphedPaddedDecodeStep._clone_token_pool_decode_context(context)
        self.assertIsNotNone(cloned)
        self.assertIs(
            cloned.metadata_by_layer_id[0],
            cloned.metadata_by_layer_type["sliding_attention"],
        )
        self.assertIs(
            cloned.metadata_by_layer_id[1],
            cloned.metadata_by_layer_type["sliding_attention"],
        )
        self.assertIsNot(cloned.paged_metadata_by_layer_type["sliding_attention"], paged)
        self.assertIs(
            cloned.paged_metadata_by_layer_id[0],
            cloned.paged_metadata_by_layer_type["sliding_attention"],
        )
        self.assertIs(
            cloned.paged_metadata_by_layer_id[1],
            cloned.paged_metadata_by_layer_type["sliding_attention"],
        )
        self.assertEqual(
            cloned.paged_metadata_by_layer_type["sliding_attention"].block_tables.tolist(),
            [[0]],
        )

        updated_flat = build_decode_metadata_from_token_slot_rows(
            [[4, 5]],
            logical_seq_lens=[6],
            out_cache_loc=[5],
        )
        updated = build_paged_decode_metadata_from_token_slot_rows(
            [[4, 5]],
            block_size=4,
            logical_seq_lens=[6],
            selected_start_positions=[4],
            out_cache_loc=[5],
        )
        copy_stats: dict[str, int] = {}
        copied_pairs: set[tuple[int, int]] = set()
        _GraphedPaddedDecodeStep._copy_decode_metadata_group(
            {
                "by_type": cloned.metadata_by_layer_type["sliding_attention"],
                "layer_0": cloned.metadata_by_layer_id[0],
                "layer_1": cloned.metadata_by_layer_id[1],
            },
            {
                "by_type": updated_flat,
                "layer_0": updated_flat,
                "layer_1": updated_flat,
            },
            "metadata",
            copied=copied_pairs,
            stats=copy_stats,
        )
        _GraphedPaddedDecodeStep._copy_decode_metadata_group(
            {
                "by_type": cloned.paged_metadata_by_layer_type["sliding_attention"],
                "layer_0": cloned.paged_metadata_by_layer_id[0],
                "layer_1": cloned.paged_metadata_by_layer_id[1],
            },
            {
                "by_type": updated,
                "layer_0": updated,
                "layer_1": updated,
            },
            "paged_metadata",
            copied=copied_pairs,
            stats=copy_stats,
        )
        self.assertGreater(copy_stats["cuda_graph_metadata_tensor_copies"], 0)
        self.assertGreater(copy_stats["cuda_graph_metadata_tensor_copy_skips"], 0)
        copied = cloned.paged_metadata_by_layer_type["sliding_attention"]
        self.assertEqual(copied.block_tables.tolist(), [[1]])
        self.assertEqual(copied.selected_start_positions.tolist(), [4])
        self.assertEqual(copied.out_cache_loc_long.tolist(), [5])

        mismatched_block_size = build_paged_decode_metadata_from_token_slot_rows(
            [[8, 9]],
            block_size=8,
            logical_seq_lens=[10],
            selected_start_positions=[8],
            out_cache_loc=[9],
        )
        with self.assertRaisesRegex(ValueError, "block_size changed"):
            _GraphedPaddedDecodeStep._copy_decode_metadata(
                copied,
                mismatched_block_size,
                "paged_metadata_by_layer_type.sliding_attention",
            )

    def test_graph_buffer_aliases_workspace_metadata_and_skips_copies(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenPoolDecodeContext,
            TokenPoolDecodeGraphBuffer,
        )

        table = ReqToTokenTable(max_requests=1, max_context_len=8)
        slot = table.allocate("a")
        table.append_slots(slot, [0, 1])
        page_table = torch.full((1, 2), -1, dtype=torch.int32)
        page_table[slot, 0] = 0
        flat = table.build_decode_metadata(
            [slot],
            out_cache_loc=[1],
            workspace_key="graph_flat",
        )
        paged = table.build_paged_decode_metadata_from_page_table_tensor(
            [slot],
            page_table,
            block_size=2,
            block_table_width=1,
            out_cache_loc=[1],
            workspace_key="graph_paged",
        )
        context = TokenPoolDecodeContext(
            metadata_by_layer_type={"sliding_attention": flat},
            metadata_by_layer_id={0: flat},
            paged_metadata_by_layer_type={"sliding_attention": paged},
            paged_metadata_by_layer_id={0: paged},
            kv_pool=object(),
            covered_layer_types=frozenset({"sliding_attention"}),
        )

        buffer = TokenPoolDecodeGraphBuffer.capture(context, clone_tensors=False)
        self.assertIs(buffer.context.metadata_by_layer_id[0], flat)
        self.assertIs(
            buffer.context.paged_metadata_by_layer_id[0],
            paged,
        )
        self.assertEqual(buffer.context.metadata_by_layer_type["sliding_attention"].kv_indices.tolist(), [0, 1])
        self.assertEqual(buffer.context.paged_metadata_by_layer_type["sliding_attention"].block_tables.tolist(), [[0]])

        table.req_to_token[slot, :2] = torch.tensor([4, 5], dtype=torch.int32)
        page_table[slot, 0] = 2
        updated_flat = table.build_decode_metadata(
            [slot],
            out_cache_loc=[5],
            workspace_key="graph_flat",
        )
        updated_paged = table.build_paged_decode_metadata_from_page_table_tensor(
            [slot],
            page_table,
            block_size=2,
            block_table_width=1,
            out_cache_loc=[5],
            workspace_key="graph_paged",
            validate=False,
        )
        updated = TokenPoolDecodeContext(
            metadata_by_layer_type={"sliding_attention": updated_flat},
            metadata_by_layer_id={0: updated_flat},
            paged_metadata_by_layer_type={"sliding_attention": updated_paged},
            paged_metadata_by_layer_id={0: updated_paged},
            kv_pool=context.kv_pool,
            covered_layer_types=context.covered_layer_types,
        )

        stats = buffer.copy_from(updated)
        self.assertEqual(stats["cuda_graph_metadata_tensor_copies"], 0)
        self.assertGreater(stats["cuda_graph_metadata_tensor_copy_skips"], 0)
        self.assertEqual(
            buffer.context.metadata_by_layer_type["sliding_attention"].kv_indices.tolist(),
            [4, 5],
        )
        self.assertEqual(
            buffer.context.paged_metadata_by_layer_type["sliding_attention"].block_tables.tolist(),
            [[2]],
        )
        self.assertEqual(
            buffer.context.paged_metadata_by_layer_type["sliding_attention"].out_cache_loc_long.tolist(),
            [5],
        )

    def test_decode_metadata_owns_triton_decode_plan(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        import os

        from wkvm.runner.gemma_token_pool import (
            build_decode_metadata_from_token_slot_rows,
            build_paged_decode_metadata_from_token_slot_rows,
        )

        old_split = os.environ.get("WKVM_TOKEN_POOL_TRITON_SPLIT_SIZE")
        old_min = os.environ.get("WKVM_TOKEN_POOL_TRITON_MIN_SPLITS")
        try:
            os.environ["WKVM_TOKEN_POOL_TRITON_SPLIT_SIZE"] = "4"
            os.environ["WKVM_TOKEN_POOL_TRITON_MIN_SPLITS"] = "3"

            flat = build_decode_metadata_from_token_slot_rows(
                [[0, 1, 2, 3, 4, 5, 6, 7, 8]],
                out_cache_loc=[8],
            )
            self.assertEqual(flat.max_seq_len, 9)
            self.assertTrue(flat.triton_decode_plan.should_split)
            self.assertEqual(flat.triton_decode_plan.split_size, 4)
            self.assertEqual(flat.triton_decode_plan.min_splits, 3)
            self.assertEqual(flat.triton_decode_plan.max_splits, 3)

            paged = build_paged_decode_metadata_from_token_slot_rows(
                [[0, 1, 2, 3, 4]],
                block_size=2,
                out_cache_loc=[4],
            )
            self.assertEqual(paged.max_seq_len, 5)
            self.assertFalse(paged.triton_decode_plan.should_split)
            self.assertEqual(paged.triton_decode_plan.split_size, 4)
            self.assertEqual(paged.triton_decode_plan.min_splits, 3)
            self.assertEqual(paged.triton_decode_plan.max_splits, 2)
        finally:
            if old_split is None:
                os.environ.pop("WKVM_TOKEN_POOL_TRITON_SPLIT_SIZE", None)
            else:
                os.environ["WKVM_TOKEN_POOL_TRITON_SPLIT_SIZE"] = old_split
            if old_min is None:
                os.environ.pop("WKVM_TOKEN_POOL_TRITON_MIN_SPLITS", None)
            else:
                os.environ["WKVM_TOKEN_POOL_TRITON_MIN_SPLITS"] = old_min

    def test_graph_copy_validates_scalar_triton_decode_plan(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        import os

        from wkvm.runner.gemma_token_pool import (
            build_decode_metadata_from_token_slot_rows,
            TokenPoolDecodeGraphBuffer,
        )

        old_split = os.environ.get("WKVM_TOKEN_POOL_TRITON_SPLIT_SIZE")
        old_min = os.environ.get("WKVM_TOKEN_POOL_TRITON_MIN_SPLITS")
        try:
            os.environ["WKVM_TOKEN_POOL_TRITON_SPLIT_SIZE"] = "2"
            os.environ["WKVM_TOKEN_POOL_TRITON_MIN_SPLITS"] = "2"
            original = build_decode_metadata_from_token_slot_rows(
                [[0, 1, 2, 3]],
                out_cache_loc=[3],
            )
            captured = TokenPoolDecodeGraphBuffer._clone_decode_metadata(original)

            os.environ["WKVM_TOKEN_POOL_TRITON_SPLIT_SIZE"] = "4"
            updated = build_decode_metadata_from_token_slot_rows(
                [[4, 5, 6, 7]],
                out_cache_loc=[7],
            )
            with self.assertRaisesRegex(ValueError, "triton_decode_plan changed"):
                TokenPoolDecodeGraphBuffer._copy_decode_metadata(
                    captured,
                    updated,
                    "metadata_by_layer_type.sliding_attention",
                )

            longer = build_decode_metadata_from_token_slot_rows(
                [[4, 5, 6, 7, 8]],
                out_cache_loc=[8],
            )
            with self.assertRaisesRegex(ValueError, "max_seq_len changed"):
                TokenPoolDecodeGraphBuffer._copy_decode_metadata(
                    captured,
                    longer,
                    "metadata_by_layer_type.sliding_attention",
                )
        finally:
            if old_split is None:
                os.environ.pop("WKVM_TOKEN_POOL_TRITON_SPLIT_SIZE", None)
            else:
                os.environ["WKVM_TOKEN_POOL_TRITON_SPLIT_SIZE"] = old_split
            if old_min is None:
                os.environ.pop("WKVM_TOKEN_POOL_TRITON_MIN_SPLITS", None)
            else:
                os.environ["WKVM_TOKEN_POOL_TRITON_MIN_SPLITS"] = old_min

    def test_graph_buffer_aliases_explicit_row_workspace_metadata(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            build_decode_metadata_from_token_slot_rows,
            TokenPoolDecodeContext,
            TokenPoolDecodeGraphBuffer,
        )

        workspace = {}
        full = build_decode_metadata_from_token_slot_rows(
            [[10, 2, 3]],
            req_slots=[7],
            logical_seq_lens=[99],
            out_cache_loc=[3],
            workspace=workspace,
            kv_indices_padding_slots=2,
        )
        context = TokenPoolDecodeContext(
            metadata_by_layer_type={"full_attention": full},
            metadata_by_layer_id={1: full},
            kv_pool=object(),
            covered_layer_types=frozenset({"full_attention"}),
            layer_id_metadata_only_types=frozenset({"full_attention"}),
        )

        buffer = TokenPoolDecodeGraphBuffer.capture(context, clone_tensors=False)
        self.assertIs(buffer.context.metadata_by_layer_type["full_attention"], full)
        self.assertEqual(
            buffer.context.metadata_by_layer_type["full_attention"].kv_indices.tolist(),
            [10, 2, 3, 3, 3],
        )

        updated_full = build_decode_metadata_from_token_slot_rows(
            [[11, 4, 5]],
            req_slots=[8],
            logical_seq_lens=[100],
            out_cache_loc=[5],
            workspace=workspace,
            kv_indices_padding_slots=2,
        )
        updated = TokenPoolDecodeContext(
            metadata_by_layer_type={"full_attention": updated_full},
            metadata_by_layer_id={1: updated_full},
            kv_pool=context.kv_pool,
            covered_layer_types=context.covered_layer_types,
            layer_id_metadata_only_types=context.layer_id_metadata_only_types,
        )

        stats = buffer.copy_from(updated)
        self.assertEqual(stats["cuda_graph_metadata_tensor_copies"], 0)
        self.assertGreater(stats["cuda_graph_metadata_tensor_copy_skips"], 0)
        self.assertEqual(
            buffer.context.metadata_by_layer_type["full_attention"].kv_indices.tolist(),
            [11, 4, 5, 5, 5],
        )
        self.assertEqual(
            buffer.context.metadata_by_layer_type["full_attention"].logical_seq_lens.tolist(),
            [100],
        )

    def test_graph_metadata_pair_memo_skips_repeated_alias_copy(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            build_decode_metadata_from_token_slot_rows,
            TokenPoolDecodeGraphBuffer,
        )

        original = build_decode_metadata_from_token_slot_rows(
            [[1, 2]],
            out_cache_loc=[2],
        )
        updated = build_decode_metadata_from_token_slot_rows(
            [[3, 4]],
            out_cache_loc=[4],
        )
        captured = TokenPoolDecodeGraphBuffer._clone_decode_metadata(original)
        copied_metadata = {(id(captured), id(updated))}
        stats = {
            "cuda_graph_metadata_tensor_copies": 0,
            "cuda_graph_metadata_tensor_copy_skips": 0,
        }

        TokenPoolDecodeGraphBuffer._copy_decode_metadata(
            captured,
            updated,
            "metadata_by_layer_type.full_attention",
            copied=set(),
            copied_metadata=copied_metadata,
            stats=stats,
        )

        self.assertEqual(stats["cuda_graph_metadata_tensor_copies"], 0)
        self.assertEqual(
            stats["cuda_graph_metadata_tensor_copy_skips"],
            TokenPoolDecodeGraphBuffer._decode_metadata_tensor_pair_count(
                captured,
                updated,
            ),
        )
        self.assertEqual(captured.kv_indices.tolist(), [1, 2])
        self.assertEqual(captured.out_cache_loc.tolist(), [2])

    def test_req_to_token_table_grows_and_rolls_back(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import ReqToTokenTable

        table = ReqToTokenTable(max_requests=1, max_context_len=2)
        slot = table.allocate("a")
        table.append_slots(slot, [1, 2])
        table.ensure_context_len(5)
        table.append_slots(slot, [3, 4, 5])
        self.assertGreaterEqual(table.max_context_len, 5)
        self.assertEqual(table.slots_for(slot).tolist(), [1, 2, 3, 4, 5])

        table.truncate(slot, 2)
        self.assertEqual(table.length(slot), 2)
        self.assertEqual(table.slots_for("a").tolist(), [1, 2])
        self.assertTrue((table.req_to_token[slot, 2:] == table.padding_token).all().item())
        with self.assertRaisesRegex(ValueError, "within"):
            table.truncate(slot, 3)

    def test_token_slot_allocator_reuses_and_rejects_overflow(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import TokenSlotAllocator

        alloc = TokenSlotAllocator(capacity=3)
        a, a_ids = alloc.alloc_slots_with_ids(2)
        self.assertEqual(a.tolist(), [0, 1])
        self.assertEqual(a_ids, [0, 1])
        self.assertEqual(alloc.allocated_count, 2)
        self.assertEqual(alloc.high_watermark, 2)
        alloc.free_slots(a[:1])
        b = alloc.alloc_slots(2)
        self.assertEqual(b.tolist(), [0, 2])
        self.assertEqual(alloc.allocated_count, 3)
        self.assertEqual(alloc.high_watermark, 3)
        with self.assertRaisesRegex(RuntimeError, "capacity"):
            alloc.alloc_slots(1)
        alloc.free_slots(torch.tensor([0, 1, 2], dtype=torch.int32))
        self.assertEqual(alloc.allocated_count, 0)

    def test_token_slot_allocator_allocates_whole_page_blocks(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import TokenSlotAllocator

        alloc = TokenSlotAllocator(capacity=16)
        first_block, first_slots = alloc.alloc_page_block_with_ids(4)
        self.assertEqual(first_block, 0)
        self.assertEqual(first_slots, [0, 1, 2, 3])
        token_slots, token_slot_ids = alloc.alloc_slots_with_ids(2)
        self.assertEqual(token_slot_ids, [4, 5])
        self.assertEqual(token_slots.tolist(), [4, 5])
        second_block, second_slots = alloc.alloc_page_block_with_ids(4)
        self.assertEqual(second_block, 2)
        self.assertEqual(second_slots, [8, 9, 10, 11])

    def test_token_kv_pool_alloc_write_gather_and_free(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import TokenKVLayerSpec, TokenKVPool

        pool = TokenKVPool(
            capacity=4,
            layer_specs=[
                TokenKVLayerSpec(layer_id=0, num_kv_heads=2, head_dim=3, dtype=torch.float32),
                TokenKVLayerSpec(layer_id=1, num_kv_heads=1, head_dim=2, dtype=torch.float32),
            ],
            dtype=torch.float32,
        )
        slots, slot_ids = pool.alloc_slots_with_ids(2)
        self.assertEqual(slots.tolist(), [0, 1])
        self.assertEqual(slot_ids, [0, 1])

        key = torch.arange(12, dtype=torch.float32).reshape(2, 2, 1, 3)
        value = key + 100
        pool.set_kv(0, slots, key, value)
        gathered_k, gathered_v = pool.gather_kv(0, slots.flip(0))
        self.assertTrue(torch.equal(gathered_k, key[:, :, 0, :].flip(0)))
        self.assertTrue(torch.equal(gathered_v, value[:, :, 0, :].flip(0)))

        self.assertGreater(pool.state_bytes(), 0)
        pool.free_slots(slots)
        reused = pool.alloc_slots(2)
        self.assertEqual(reused.tolist(), [0, 1])

    def test_token_kv_pool_can_defer_layer_buffer_allocation(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import TokenKVLayerSpec, TokenKVPool

        pool = TokenKVPool(
            capacity=4,
            layer_specs=[
                TokenKVLayerSpec(layer_id=0, num_kv_heads=2, head_dim=3, dtype=torch.float32),
                TokenKVLayerSpec(layer_id=1, num_kv_heads=1, head_dim=2, dtype=torch.float32),
            ],
            dtype=torch.float32,
            defer_buffer_allocation=True,
        )
        self.assertEqual(pool.state_bytes(), 0)
        self.assertEqual(pool.allocated_layer_count, 0)

        slots = pool.alloc_slots(2)
        key0 = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
        value0 = key0 + 100
        pool.set_kv(0, slots, key0, value0)
        first_bytes = pool.state_bytes()
        self.assertGreater(first_bytes, 0)
        self.assertEqual(pool.allocated_layer_count, 1)

        key1 = torch.arange(4, dtype=torch.float32).reshape(2, 1, 2)
        value1 = key1 + 200
        pool.set_kv(1, slots, key1, value1)
        self.assertGreater(pool.state_bytes(), first_bytes)
        self.assertEqual(pool.allocated_layer_count, 2)

    def test_token_kv_pool_can_disable_slot_write_validation(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import TokenKVLayerSpec, TokenKVPool

        spec = [TokenKVLayerSpec(layer_id=0, num_kv_heads=1, head_dim=2, dtype=torch.float32)]
        key = torch.ones(1, 1, 2, dtype=torch.float32)
        value = key + 10
        strict_pool = TokenKVPool(
            capacity=2,
            layer_specs=spec,
            dtype=torch.float32,
        )
        with self.assertRaisesRegex(KeyError, "not allocated"):
            strict_pool.set_kv(0, [1], key, value)

        fast_pool = TokenKVPool(
            capacity=2,
            layer_specs=spec,
            dtype=torch.float32,
            validate_slot_writes=False,
        )
        fast_pool.set_kv(0, [1], key, value)
        gathered_k, gathered_v = fast_pool.gather_kv(0, [1])
        self.assertTrue(torch.equal(gathered_k, key))
        self.assertTrue(torch.equal(gathered_v, value))

    def test_token_kv_pool_uses_slice_copy_for_host_contiguous_slots(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import TokenKVLayerSpec, TokenKVPool

        pool = TokenKVPool(
            capacity=6,
            layer_specs=[
                TokenKVLayerSpec(layer_id=0, num_kv_heads=1, head_dim=2, dtype=torch.float32),
            ],
            dtype=torch.float32,
            validate_slot_writes=False,
        )
        key = torch.arange(6, dtype=torch.float32).reshape(3, 1, 2)
        value = key + 10

        pool.set_kv(0, [2, 3, 4], key, value)

        gathered_k, gathered_v = pool.gather_kv(0, [2, 3, 4])
        self.assertTrue(torch.equal(gathered_k, key))
        self.assertTrue(torch.equal(gathered_v, value))
        self.assertEqual(pool.kv_set_calls, 1)
        self.assertEqual(pool.kv_set_tokens, 3)
        self.assertEqual(pool.kv_set_slice_copy_calls, 1)
        self.assertEqual(pool.kv_set_index_copy_calls, 0)

    def test_token_kv_pool_keeps_index_copy_for_noncontiguous_slots(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import TokenKVLayerSpec, TokenKVPool

        pool = TokenKVPool(
            capacity=6,
            layer_specs=[
                TokenKVLayerSpec(layer_id=0, num_kv_heads=1, head_dim=2, dtype=torch.float32),
            ],
            dtype=torch.float32,
            validate_slot_writes=False,
        )
        key = torch.arange(6, dtype=torch.float32).reshape(3, 1, 2)
        value = key + 10

        pool.set_kv(0, [4, 2, 5], key, value)

        gathered_k, gathered_v = pool.gather_kv(0, [4, 2, 5])
        self.assertTrue(torch.equal(gathered_k, key))
        self.assertTrue(torch.equal(gathered_v, value))
        self.assertEqual(pool.kv_set_calls, 1)
        self.assertEqual(pool.kv_set_tokens, 3)
        self.assertEqual(pool.kv_set_slice_copy_calls, 0)
        self.assertEqual(pool.kv_set_index_copy_calls, 1)

    def test_token_kv_pool_uses_triton_store_for_cuda_slot_mapped_writes(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        if not torch.cuda.is_available():
            self.skipTest("CUDA unavailable")
        try:
            from wkvm.runner.gemma_token_pool_triton import token_pool_store_kv
        except ImportError:
            self.skipTest("triton unavailable")
        self.assertIsNotNone(token_pool_store_kv)

        import os

        from wkvm.runner.gemma_token_pool import TokenKVLayerSpec, TokenKVPool

        old_enable = os.environ.get("WKVM_ENABLE_TOKEN_POOL_KV_STORE_TRITON")
        old_disable = os.environ.get("WKVM_DISABLE_TOKEN_POOL_KV_STORE_TRITON")
        old_global_disable = os.environ.get("WKVM_DISABLE_TOKEN_POOL_TRITON")
        try:
            os.environ["WKVM_ENABLE_TOKEN_POOL_KV_STORE_TRITON"] = "1"
            os.environ.pop("WKVM_DISABLE_TOKEN_POOL_KV_STORE_TRITON", None)
            os.environ.pop("WKVM_DISABLE_TOKEN_POOL_TRITON", None)
            pool = TokenKVPool(
                capacity=6,
                layer_specs=[
                    TokenKVLayerSpec(
                        layer_id=0,
                        num_kv_heads=1,
                        head_dim=2,
                        dtype=torch.float32,
                    ),
                ],
                dtype=torch.float32,
                device="cuda",
                validate_slot_writes=False,
            )
            key = torch.arange(6, dtype=torch.float32, device="cuda").reshape(3, 1, 2)
            value = key + 10
            slots = torch.tensor([4, 2, 5], dtype=torch.int32, device="cuda")

            pool.set_kv(0, slots, key, value)
            torch.cuda.synchronize()

            gathered_k, gathered_v = pool.gather_kv(0, slots)
            self.assertTrue(torch.equal(gathered_k.cpu(), key.cpu()))
            self.assertTrue(torch.equal(gathered_v.cpu(), value.cpu()))
            self.assertEqual(pool.kv_set_calls, 1)
            self.assertEqual(pool.kv_set_tokens, 3)
            self.assertEqual(pool.kv_set_slice_copy_calls, 0)
            self.assertEqual(pool.kv_set_index_copy_calls, 0)
            self.assertEqual(pool.kv_set_triton_copy_calls, 1)
            self.assertEqual(pool.kv_set_triton_fallback_calls, 0)
        finally:
            if old_enable is None:
                os.environ.pop("WKVM_ENABLE_TOKEN_POOL_KV_STORE_TRITON", None)
            else:
                os.environ["WKVM_ENABLE_TOKEN_POOL_KV_STORE_TRITON"] = old_enable
            if old_disable is None:
                os.environ.pop("WKVM_DISABLE_TOKEN_POOL_KV_STORE_TRITON", None)
            else:
                os.environ["WKVM_DISABLE_TOKEN_POOL_KV_STORE_TRITON"] = old_disable
            if old_global_disable is None:
                os.environ.pop("WKVM_DISABLE_TOKEN_POOL_TRITON", None)
            else:
                os.environ["WKVM_DISABLE_TOKEN_POOL_TRITON"] = old_global_disable

    def test_token_kv_pool_reuses_attention_output_buffer(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import TokenKVLayerSpec, TokenKVPool

        pool = TokenKVPool(
            capacity=2,
            layer_specs=[
                TokenKVLayerSpec(layer_id=0, num_kv_heads=1, head_dim=2, dtype=torch.float32),
            ],
            dtype=torch.float32,
        )
        first = pool.attention_output_buffer(
            batch=2,
            query_heads=4,
            head_dim=8,
            dtype=torch.float32,
            device="cpu",
        )
        second = pool.attention_output_buffer(
            batch=2,
            query_heads=4,
            head_dim=8,
            dtype=torch.float32,
            device="cpu",
        )
        other = pool.attention_output_buffer(
            batch=1,
            query_heads=4,
            head_dim=8,
            dtype=torch.float32,
            device="cpu",
        )

        self.assertIs(first, second)
        self.assertEqual(tuple(first.shape), (2, 1, 4, 8))
        self.assertIsNot(first, other)

    def test_token_kv_pool_aliases_shared_layers(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import TokenKVLayerSpec, TokenKVPool

        pool = TokenKVPool(
            capacity=3,
            layer_specs=[
                TokenKVLayerSpec(layer_id=0, num_kv_heads=2, head_dim=3, dtype=torch.float32),
                TokenKVLayerSpec(
                    layer_id=1,
                    num_kv_heads=2,
                    head_dim=3,
                    dtype=torch.float32,
                    kv_share_target_layer=0,
                ),
            ],
            dtype=torch.float32,
        )
        self.assertEqual(pool.target_layer(1), 0)
        self.assertIs(pool.get_kv_buffer(1)[0], pool.get_kv_buffer(0)[0])

        slots = pool.alloc_slots(1)
        key = torch.ones(1, 2, 3, dtype=torch.float32)
        value = key + 10
        pool.set_kv(0, slots, key, value)
        gathered_k, gathered_v = pool.gather_kv(1, slots)
        self.assertTrue(torch.equal(gathered_k, key))
        self.assertTrue(torch.equal(gathered_v, value))
        with self.assertRaisesRegex(ValueError, "shares KV"):
            pool.set_kv(1, slots, key, value)

    def test_token_kv_pool_validates_shared_layer_shape(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import TokenKVLayerSpec, TokenKVPool

        with self.assertRaisesRegex(ValueError, "shared KV shape"):
            TokenKVPool(
                capacity=3,
                layer_specs=[
                    TokenKVLayerSpec(layer_id=0, num_kv_heads=2, head_dim=3, dtype=torch.float32),
                    TokenKVLayerSpec(
                        layer_id=1,
                        num_kv_heads=1,
                        head_dim=3,
                        dtype=torch.float32,
                        kv_share_target_layer=0,
                    ),
                ],
                dtype=torch.float32,
            )

    def test_token_kv_pool_skips_cpu_slot_validation_during_cuda_graph_capture(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import TokenKVLayerSpec, TokenKVPool

        class CaptureSlots:
            is_cuda = True

            def detach(self):
                raise AssertionError("capture path must not copy slots to CPU")

        pool = TokenKVPool(
            capacity=3,
            layer_specs=[
                TokenKVLayerSpec(layer_id=0, num_kv_heads=1, head_dim=2, dtype=torch.float32),
            ],
            dtype=torch.float32,
        )
        with unittest.mock.patch.object(torch.cuda, "is_available", return_value=True):
            with unittest.mock.patch.object(
                torch.cuda,
                "is_current_stream_capturing",
                return_value=True,
            ):
                pool._validate_allocated_token_slots(CaptureSlots())


if __name__ == "__main__":
    unittest.main()
