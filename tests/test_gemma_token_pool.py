import unittest


class TestGemmaTokenPool(unittest.TestCase):
    def test_decode_reservation_is_token_pool_owned_state(self) -> None:
        from wkvm.runner.gemma_token_pool import (
            TokenPoolDecodeReservation,
            TokenPoolRequestPageStateSnapshot,
        )

        snapshot = TokenPoolRequestPageStateSnapshot(
            req_id="req",
            req_slot=3,
            page_table={0: 4},
            owned_slots=frozenset({16, 17}),
        )
        reservation = TokenPoolDecodeReservation(
            req_id="req",
            req_slot=3,
            token_slot=18,
            token_slot_tensor="slot_tensor",
            previous_length=7,
            persistent_full_attention_row=True,
            page_state_snapshot=snapshot,
        )

        reservation.full_attention_token_slot = 99

        self.assertEqual(reservation.req_id, "req")
        self.assertEqual(reservation.req_slot, 3)
        self.assertEqual(reservation.token_slot, 18)
        self.assertEqual(reservation.previous_length, 7)
        self.assertTrue(reservation.persistent_full_attention_row)
        self.assertIs(reservation.page_state_snapshot, snapshot)
        self.assertEqual(reservation.full_attention_token_slot, 99)

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

    def test_token_pool_block_tables_stage_gather_and_slot_mapping(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import TokenPoolBlockTables

        tables = TokenPoolBlockTables(
            max_requests=2,
            max_context_len=4,
            block_size=4,
        )
        self.assertEqual(tables.shape, (2, 1))
        tables.stage_block(0, 0, 3)
        self.assertEqual(tables.block_for(0, 0), -1)
        self.assertEqual(tables.apply_staged_writes(), 1)
        self.assertEqual(tables.block_for(0, 0), 3)

        tables.set_block(0, 1, 5)
        tables.set_block(1, 2, 8)
        self.assertEqual(tables.shape, (2, 3))
        gathered = tables.gather_block_tables(
            [0, 1],
            [0, 2],
            [2, 1],
            block_table_width=3,
            workspace_key="decode",
        )
        gathered_ptr = int(gathered.data_ptr())
        self.assertEqual(gathered.tolist(), [[3, 5, -1], [8, -1, -1]])

        tables.set_block(0, 1, 6)
        gathered_again = tables.gather_block_tables(
            [0, 1],
            [0, 2],
            [2, 1],
            block_table_width=3,
            workspace_key="decode",
        )
        self.assertEqual(int(gathered_again.data_ptr()), gathered_ptr)
        self.assertEqual(gathered_again.tolist(), [[3, 6, -1], [8, -1, -1]])

        slots = tables.compute_slot_mapping(
            [0, 1],
            [5, 8],
            workspace_key="decode",
        )
        slots_ptr = int(slots.data_ptr())
        self.assertEqual(slots.tolist(), [25, 32])
        tables.clear_block(1, 2)
        slots_again = tables.compute_slot_mapping(
            [0, 1],
            [5, 8],
            pad_slot_id=-99,
            workspace_key="decode",
        )
        self.assertEqual(int(slots_again.data_ptr()), slots_ptr)
        self.assertEqual(slots_again.tolist(), [25, -99])
        self.assertEqual(slots_again.dtype, torch.long)
        self.assertGreater(tables.state_bytes(), tables.tensor.numel() * 4)

    def test_token_pool_block_tables_snapshot_restore_row(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import TokenPoolBlockTables

        tables = TokenPoolBlockTables(
            max_requests=1,
            max_context_len=4,
            block_size=4,
        )
        tables.set_block(0, 0, 2)
        snapshot = tables.snapshot_row(0)
        tables.set_block(0, 2, 7)
        self.assertEqual(tables.tensor.tolist(), [[2, -1, 7]])
        tables.restore_row(0, snapshot)
        self.assertEqual(tables.tensor.tolist(), [[2, -1, -1]])
        tables.reset_row(0)
        self.assertEqual(tables.tensor.tolist(), [[-1, -1, -1]])

    def test_decode_backend_wraps_page_table_lifecycle(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenPoolBlockTables,
            TokenPoolDecodeBackendState,
        )

        metadata_only = TokenPoolDecodeBackendState(
            table=ReqToTokenTable(max_requests=1, max_context_len=4),
            block_size=4,
        )
        self.assertIsNone(metadata_only.page_table_tensor)
        self.assertIsNone(metadata_only.snapshot_page_table_row(0))
        metadata_only.ensure_page_table_width(32)
        metadata_only.reset_page_table_row(0)
        metadata_only.restore_page_table_row(0, None)
        metadata_only.set_page_table_block(0, 0, 1)
        metadata_only.clear_page_table_block(0, 0)

        block_tables = TokenPoolBlockTables(
            max_requests=1,
            max_context_len=4,
            block_size=4,
        )
        backend = TokenPoolDecodeBackendState(
            table=ReqToTokenTable(max_requests=1, max_context_len=4),
            block_tables=block_tables,
            block_size=4,
        )
        self.assertIs(backend.page_table_tensor, block_tables.tensor)
        backend.set_page_table_block(0, 0, 2)
        snapshot = backend.snapshot_page_table_row(0)
        backend.set_page_table_block(0, 2, 7)
        self.assertEqual(block_tables.tensor.tolist(), [[2, -1, 7]])
        backend.restore_page_table_row(0, snapshot)
        self.assertEqual(block_tables.tensor.tolist(), [[2, -1, -1]])
        backend.ensure_page_table_width(17)
        self.assertEqual(block_tables.shape, (1, 5))
        backend.clear_page_table_block(0, 0)
        self.assertEqual(block_tables.tensor[0, :3].tolist(), [-1, -1, -1])
        backend.set_page_table_block(0, 1, 3)
        backend.reset_page_table_row(0)
        self.assertEqual(block_tables.tensor[0, :3].tolist(), [-1, -1, -1])

    def test_decode_backend_owns_request_slot_lifecycle(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenPoolBlockTables,
            TokenPoolDecodeBackendState,
            TokenPoolFullAttentionRow,
            TokenSlotAllocator,
        )

        table = ReqToTokenTable(max_requests=1, max_context_len=8)
        block_tables = TokenPoolBlockTables(
            max_requests=1,
            max_context_len=8,
            block_size=4,
        )
        allocator = TokenSlotAllocator(capacity=16)
        backend = TokenPoolDecodeBackendState(
            table=table,
            allocator=allocator,
            block_tables=block_tables,
            block_size=4,
        )

        req_slot = backend.admit_request("req")
        self.assertEqual(req_slot, 0)
        self.assertTrue(backend.has_request("req"))
        self.assertEqual(backend.request_slot_for("req"), req_slot)
        self.assertEqual(backend.active_request_slots, 1)
        self.assertEqual(backend.request_slots, {"req": 0})
        self.assertEqual(backend.request_token_slots, {"req": []})
        self.assertEqual(backend.admit_request("req"), req_slot)
        backend.ensure_context_len(9)
        self.assertGreaterEqual(table.max_context_len, 9)
        self.assertGreaterEqual(block_tables.shape[1], 3)
        backend.append_table_slots(req_slot, [9, 10, 11])
        self.assertEqual(backend.request_length(req_slot), 3)
        self.assertEqual(backend.clear_table_before(req_slot, 2), [9, 10])
        backend.truncate_table_row(req_slot, 1)
        self.assertEqual(backend.request_length("req"), 1)

        _, normal_token_slots = allocator.alloc_slots_with_ids(2)
        backend.append_request_token_slots("req", normal_token_slots)
        backend.allocate_page_aligned_slots("req", 0, 1, req_slot=req_slot)
        _, full_row_slots = allocator.alloc_slots_with_ids(2)
        full_attention_rows = backend.full_attention_row_records
        self.assertIsNotNone(full_attention_rows)
        full_attention_rows["req"] = TokenPoolFullAttentionRow(
            row_slots=full_row_slots,
            owned_slots=full_row_slots,
        )
        self.assertEqual(allocator.allocated_count, 8)
        released_req_slot, released_page_slots, released_token_slots = (
            backend.release_request("req")
        )

        self.assertEqual(released_req_slot, req_slot)
        self.assertEqual(released_page_slots, {4, 5, 6, 7})
        self.assertEqual(released_token_slots, [0, 1])
        self.assertFalse(backend.has_request("req"))
        self.assertEqual(backend.active_request_slots, 0)
        self.assertEqual(allocator.allocated_count, 0)
        self.assertEqual(block_tables.tensor[0].tolist(), [-1] * block_tables.shape[1])
        with self.assertRaises(KeyError):
            table.slot_for("req")

    def test_decode_backend_owns_request_page_state_rollback_and_release(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenPoolBlockTables,
            TokenPoolDecodeBackendState,
            TokenSlotAllocator,
        )

        table = ReqToTokenTable(max_requests=1, max_context_len=8)
        req_slot = table.allocate("req")
        block_tables = TokenPoolBlockTables(
            max_requests=1,
            max_context_len=8,
            block_size=4,
        )
        allocator = TokenSlotAllocator(capacity=16)
        backend = TokenPoolDecodeBackendState(
            table=table,
            allocator=allocator,
            block_tables=block_tables,
            block_size=4,
        )
        backend.admit_request_page_state("req", req_slot)

        snapshot = backend.snapshot_request_page_state("req", req_slot)
        token_slots, token_slot_ids = backend.allocate_page_aligned_slots(
            "req",
            2,
            4,
            req_slot=req_slot,
        )
        self.assertEqual(token_slots.tolist(), [2, 3, 4, 5])
        self.assertEqual(token_slot_ids, [2, 3, 4, 5])
        self.assertEqual(allocator.allocated_count, 8)
        self.assertEqual(backend.page_table_for_request("req"), {0: 0, 1: 1})
        self.assertEqual(block_tables.tensor.tolist(), [[0, 1]])

        restored = backend.restore_request_page_state(snapshot)
        self.assertEqual(restored, list(range(8)))
        self.assertEqual(allocator.allocated_count, 0)
        self.assertEqual(backend.page_table_for_request("req"), {})
        self.assertEqual(backend.page_owned_slots_for_request("req"), set())
        self.assertEqual(block_tables.tensor.tolist(), [[-1, -1]])

        token_slots, token_slot_ids = backend.allocate_page_aligned_slots(
            "req",
            0,
            1,
            req_slot=req_slot,
        )
        self.assertEqual(token_slots.tolist(), [0])
        self.assertEqual(token_slot_ids, [0])
        self.assertEqual(allocator.allocated_count, 4)
        released = backend.release_request_page_state("req", req_slot)
        self.assertEqual(released, {0, 1, 2, 3})
        self.assertEqual(allocator.allocated_count, 0)
        self.assertEqual(block_tables.tensor.tolist(), [[-1, -1]])

    def test_decode_backend_releases_expired_page_blocks(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenKVLayerSpec,
            TokenKVPool,
            TokenPoolBlockTables,
            TokenPoolDecodeBackendState,
        )

        table = ReqToTokenTable(max_requests=1, max_context_len=8)
        req_slot = table.allocate("req")
        block_tables = TokenPoolBlockTables(
            max_requests=1,
            max_context_len=8,
            block_size=4,
        )
        pool = TokenKVPool(
            capacity=16,
            layer_specs=[TokenKVLayerSpec(layer_id=0, num_kv_heads=1, head_dim=2)],
            defer_buffer_allocation=True,
        )
        backend = TokenPoolDecodeBackendState(
            table=table,
            allocator=pool,
            kv_pool=pool,
            block_tables=block_tables,
            block_size=4,
        )
        backend.admit_request_page_state("req", req_slot)
        backend.allocate_page_aligned_slots("req", 0, 8, req_slot=req_slot)

        freed = backend.release_expired_page_blocks("req", req_slot, 4)

        self.assertEqual(freed, [0, 1, 2, 3])
        self.assertEqual(pool.allocated_count, 4)
        self.assertEqual(backend.page_table_for_request("req"), {1: 1})
        self.assertEqual(backend.page_owned_slots_for_request("req"), {4, 5, 6, 7})
        self.assertEqual(block_tables.tensor.tolist(), [[-1, 1]])

        backend.release_request_page_state("req", req_slot)
        self.assertEqual(pool.allocated_count, 0)

    def test_decode_backend_releases_dropped_table_slots_around_page_blocks(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenPoolBlockTables,
            TokenPoolDecodeBackendState,
            TokenSlotAllocator,
        )

        table = ReqToTokenTable(max_requests=1, max_context_len=8)
        block_tables = TokenPoolBlockTables(
            max_requests=1,
            max_context_len=8,
            block_size=4,
        )
        allocator = TokenSlotAllocator(capacity=16)
        backend = TokenPoolDecodeBackendState(
            table=table,
            allocator=allocator,
            block_tables=block_tables,
            block_size=4,
        )
        req_slot = backend.admit_request("req")
        _, normal_slots = allocator.alloc_slots_with_ids(2)
        backend.append_request_token_slots("req", normal_slots)
        _, page_slot_ids = backend.allocate_page_aligned_slots(
            "req",
            0,
            1,
            req_slot=req_slot,
        )

        released = backend.release_dropped_table_slots(
            "req",
            [normal_slots[0], page_slot_ids[0]],
        )

        self.assertEqual(released, [normal_slots[0]])
        self.assertEqual(allocator.allocated_count, 5)
        self.assertEqual(backend.request_token_slots["req"], [normal_slots[1]])
        self.assertEqual(backend.page_owned_slots_for_request("req"), {4, 5, 6, 7})

        backend.release_request("req")
        self.assertEqual(allocator.allocated_count, 0)

    def test_decode_backend_clears_request_prefix_lifecycle(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenKVLayerSpec,
            TokenKVPool,
            TokenPoolBlockTables,
            TokenPoolDecodeBackendState,
        )

        table = ReqToTokenTable(max_requests=1, max_context_len=8)
        block_tables = TokenPoolBlockTables(
            max_requests=1,
            max_context_len=8,
            block_size=4,
        )
        pool = TokenKVPool(
            capacity=16,
            layer_specs=[TokenKVLayerSpec(layer_id=0, num_kv_heads=1, head_dim=2)],
            defer_buffer_allocation=True,
        )
        backend = TokenPoolDecodeBackendState(
            table=table,
            allocator=pool,
            kv_pool=pool,
            block_tables=block_tables,
            block_size=4,
        )
        req_slot = backend.admit_request("req")
        _, normal_slots = pool.alloc_slots_with_ids(2)
        backend.append_request_token_slots("req", normal_slots)
        _, page_slot_ids = backend.allocate_page_aligned_slots(
            "req",
            2,
            1,
            req_slot=req_slot,
        )
        page_owned = sorted(backend.page_owned_slots_for_request("req"))
        backend.append_table_slots(
            req_slot,
            [normal_slots[0], normal_slots[1], page_slot_ids[0], table.padding_token],
        )

        result = backend.clear_request_prefix("req", req_slot, 4)

        self.assertEqual(
            result.dropped_slots,
            (normal_slots[0], normal_slots[1], page_slot_ids[0]),
        )
        self.assertEqual(result.released_slots, tuple(normal_slots))
        self.assertEqual(result.expired_page_slots, tuple(page_owned))
        self.assertEqual(result.invalidated_full_attention_rows, 0)
        self.assertEqual(backend.request_token_slots["req"], [])
        self.assertEqual(backend.page_table_for_request("req"), {})
        self.assertEqual(backend.page_owned_slots_for_request("req"), set())
        self.assertEqual(block_tables.tensor[req_slot].tolist(), [-1, -1])
        self.assertEqual(pool.allocated_count, 0)

    def test_full_attention_row_manager_reuses_and_frees_slots(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            TokenPoolFullAttentionRowManager,
            TokenSlotAllocator,
        )

        allocator = TokenSlotAllocator(capacity=16)
        manager = TokenPoolFullAttentionRowManager(
            allocator=allocator,
            block_size=4,
        )
        materialized_slots, materialized_slot_ids = allocator.alloc_slots_with_ids(2)

        first = manager.start_persistent_row(
            "persist",
            materialized_slots=materialized_slots,
            append_reserve_slots=2,
            page_aligned=False,
        )

        self.assertFalse(first.reused_existing_row)
        self.assertEqual(first.full_token_slot, 2)
        self.assertEqual(first.row.row_slots, [0, 1, 2])
        self.assertEqual(first.row.owned_slots, [0, 1, 2, 3])
        self.assertEqual(first.row.append_slots, [3])
        self.assertEqual(materialized_slot_ids, [0, 1])

        appended = manager.append_existing_row(
            "persist",
            append_reserve_slots=2,
        )
        self.assertIsNotNone(appended)
        self.assertTrue(appended.reused_existing_row)
        self.assertIs(appended.row, first.row)
        self.assertEqual(appended.full_token_slot, 3)
        self.assertEqual(first.row.row_slots, [0, 1, 2, 3])
        self.assertEqual(first.row.append_slots, [])
        self.assertEqual(allocator.allocated_count, 4)

        self.assertEqual(manager.invalidate_containing([3, 9]), 1)
        self.assertEqual(manager.rows, {})
        self.assertEqual(allocator.allocated_count, 0)

    def test_full_attention_row_manager_page_aligned_appends_at_boundaries(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            TokenPoolFullAttentionRowManager,
            TokenSlotAllocator,
        )

        allocator = TokenSlotAllocator(capacity=32)
        manager = TokenPoolFullAttentionRowManager(
            allocator=allocator,
            block_size=4,
        )

        materialized_slots, materialized_slot_ids, first = (
            manager.start_page_aligned_persistent_row(
                "paged",
                materialized_width=3,
                append_reserve_slots=2,
            )
        )

        self.assertEqual(materialized_slots.tolist(), [0, 1, 2])
        self.assertEqual(materialized_slot_ids, [0, 1, 2])
        self.assertEqual(first.full_token_slot, 3)
        self.assertTrue(first.row.page_aligned)
        self.assertEqual(first.row.row_slots, [0, 1, 2, 3])
        self.assertEqual(first.row.append_slots, [4, 5, 6, 7])
        self.assertEqual(first.row.owned_slots, list(range(8)))

        for expected_slot in (4, 5, 6, 7):
            appended = manager.append_existing_row(
                "paged",
                append_reserve_slots=1,
            )
            self.assertIsNotNone(appended)
            self.assertEqual(appended.full_token_slot, expected_slot)

        self.assertEqual(first.row.append_slots, [])
        boundary_append = manager.append_existing_row(
            "paged",
            append_reserve_slots=1,
        )
        self.assertIsNotNone(boundary_append)
        self.assertEqual(boundary_append.full_token_slot, 8)
        self.assertEqual(first.row.append_slots, [9, 10, 11])
        self.assertEqual(first.row.owned_slots, list(range(12)))

        manager.clear(["paged"])
        self.assertEqual(manager.rows, {})
        self.assertEqual(allocator.allocated_count, 0)

    def test_full_attention_row_manager_prepares_decode_row_chunks(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            TokenPoolFullAttentionRowManager,
            TokenSlotAllocator,
        )

        allocator = TokenSlotAllocator(capacity=16)
        manager = TokenPoolFullAttentionRowManager(
            allocator=allocator,
            block_size=4,
        )
        reservation_slots, reservation_slot_ids = allocator.alloc_slots_with_ids(1)

        first = manager.prepare_decode_row(
            "persist",
            materialized_width=2,
            decode_token_slot=reservation_slot_ids[0],
            decode_token_slot_tensor=reservation_slots[:1],
            persistent_rows=True,
            build_paged_rows=False,
            append_reserve_slots=2,
            device="cpu",
        )

        self.assertFalse(first.reused_existing_row)
        self.assertTrue(first.rebuilt_persistent_row)
        self.assertEqual(first.materialized_slot_ids, [1, 2])
        self.assertEqual(first.full_token_slot, 3)
        self.assertEqual(first.row_chunks.chunks[0].tolist(), [1, 2])
        self.assertEqual(first.row_chunks.chunks[1].tolist(), [3])
        self.assertEqual(manager.rows["persist"].append_slots, [4])

        reused = manager.prepare_decode_row(
            "persist",
            materialized_width=3,
            decode_token_slot=reservation_slot_ids[0],
            decode_token_slot_tensor=reservation_slots[:1],
            persistent_rows=True,
            build_paged_rows=False,
            append_reserve_slots=2,
            device="cpu",
        )

        self.assertTrue(reused.reused_existing_row)
        self.assertTrue(reused.appended_existing_row)
        self.assertFalse(reused.rebuilt_persistent_row)
        self.assertEqual(reused.materialized_slot_ids, [])
        self.assertEqual(reused.full_token_slot, 4)
        self.assertEqual(reused.row_chunks.chunks[0].tolist(), [1, 2, 3, 4])

        manager.clear(["persist"])
        self.assertEqual(allocator.allocated_count, 1)
        allocator.free_slots(reservation_slot_ids)
        self.assertEqual(allocator.allocated_count, 0)

    def test_token_pool_decode_backend_wraps_full_attention_row_lifecycle(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenPoolDecodeBackendState,
            TokenPoolFullAttentionRow,
            TokenSlotAllocator,
        )

        metadata_only = TokenPoolDecodeBackendState(
            table=ReqToTokenTable(max_requests=1, max_context_len=4),
        )
        self.assertFalse(metadata_only.has_full_attention_rows())
        self.assertIsNone(metadata_only.full_attention_transient_slots)
        self.assertIsNone(metadata_only.full_attention_row_records)
        self.assertEqual(metadata_only.invalidate_full_attention_rows(["missing"]), 0)

        allocator = TokenSlotAllocator(capacity=24)
        backend = TokenPoolDecodeBackendState(
            table=ReqToTokenTable(max_requests=2, max_context_len=8),
            allocator=allocator,
            block_size=4,
        )
        self.assertTrue(backend.has_full_attention_rows())
        transient_slots = backend.full_attention_transient_slots
        row_records = backend.full_attention_row_records
        self.assertIsNotNone(transient_slots)
        self.assertIsNotNone(row_records)

        page_tensor, page_slots, page_owned_slots = (
            backend.allocate_page_aligned_full_attention_row_slots(0, 5)
        )
        self.assertEqual(page_tensor.tolist(), list(range(8)))
        self.assertEqual(page_slots, list(range(8)))
        self.assertEqual(page_owned_slots, list(range(8)))
        allocator.free_slots(page_owned_slots)

        row_tensor, row_owned_slots = allocator.alloc_slots_with_ids(3)
        transient_tensor, transient_owned_slots = allocator.alloc_slots_with_ids(2)
        self.assertEqual(row_tensor.tolist(), row_owned_slots)
        self.assertEqual(transient_tensor.tolist(), transient_owned_slots)
        row_records["persist"] = TokenPoolFullAttentionRow(
            row_slots=row_owned_slots[:2],
            owned_slots=row_owned_slots,
        )
        transient_slots["transient"] = transient_owned_slots
        self.assertEqual(allocator.allocated_count, 5)

        self.assertEqual(
            backend.invalidate_full_attention_rows_containing([row_owned_slots[1]]),
            1,
        )
        self.assertEqual(row_records, {})
        self.assertEqual(allocator.allocated_count, 2)
        backend.clear_full_attention_rows("transient")
        self.assertEqual(transient_slots, {})
        self.assertEqual(allocator.allocated_count, 0)
        self.assertEqual(
            backend.stats(
                active_request_slots=1,
                attention_enabled=True,
                paged_block_size=4,
            ),
            {
                "enabled": True,
                "attention_enabled": True,
                "active_request_slots": 1,
                "allocated_token_slots": 0,
                "free_token_slots": 8,
                "next_token_slot": 8,
                "token_slot_high_watermark": 8,
                "token_slot_capacity": 24,
                "paged_block_size": 4,
                "page_table_metadata_max_rows": 2,
                "max_context_len": 8,
                "metadata_bytes": 96,
                "kv_pool_bytes": 0,
                "kv_pool_layers": 0,
            },
        )

    def test_token_pool_decode_backend_builds_layer_type_metadata(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenPoolDecodeBackendState,
        )

        table = ReqToTokenTable(max_requests=2, max_context_len=8)
        a = table.allocate("a")
        b = table.allocate("b")
        table.append_slots(a, [0, 1, 2, 3, 4])
        table.append_slots(b, [8, 9, 10])
        backend = TokenPoolDecodeBackendState(table=table)

        metadata_by_type = backend.build_decode_metadata_by_layer_type(
            req_slots=[a, b],
            out_cache_loc=[4, 10],
            sliding_window=2,
        )

        self.assertEqual(set(metadata_by_type), {"full_attention", "sliding_attention"})
        full = metadata_by_type["full_attention"]
        sliding = metadata_by_type["sliding_attention"]
        self.assertEqual(full.req_pool_indices.tolist(), [a, b])
        self.assertEqual(full.logical_seq_lens.tolist(), [5, 3])
        self.assertEqual(full.seq_lens.tolist(), [5, 3])
        self.assertEqual(full.kv_indptr.tolist(), [0, 5, 8])
        self.assertEqual(full.kv_indices.tolist(), [0, 1, 2, 3, 4, 8, 9, 10])
        self.assertEqual(full.out_cache_loc.tolist(), [4, 10])
        self.assertEqual(sliding.logical_seq_lens.tolist(), [5, 3])
        self.assertEqual(sliding.seq_lens.tolist(), [2, 2])
        self.assertEqual(sliding.kv_indptr.tolist(), [0, 2, 4])
        self.assertEqual(sliding.kv_indices.tolist(), [3, 4, 9, 10])
        self.assertEqual(sliding.out_cache_loc.tolist(), [4, 10])

    def test_token_pool_decode_backend_owns_current_decode_batch_state(self) -> None:
        from types import SimpleNamespace

        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            build_decode_metadata_from_token_slot_rows,
            build_paged_decode_metadata_from_token_slot_rows,
            ReqToTokenTable,
            TokenPoolDecodeBackendState,
        )

        pool = SimpleNamespace(layer_specs={7: object()}, capacity=16)
        backend = TokenPoolDecodeBackendState(
            table=ReqToTokenTable(max_requests=1, max_context_len=8),
            kv_pool=pool,
            block_size=4,
            token_pool_capacity=16,
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

        state = backend.set_decode_batch_state(
            metadata_by_layer_type={"full_attention": by_type},
            metadata_by_layer_id={7: by_layer},
            paged_metadata_by_layer_type={"full_attention": paged_by_type},
            paged_metadata_by_layer_id={7: paged_by_layer},
            covered_layer_types={"full_attention"},
        )
        context = backend.build_current_decode_context(
            layer_id_metadata_only_types=frozenset({"full_attention"}),
        )

        self.assertIs(backend.current_decode_batch_state, state)
        self.assertEqual(
            backend.current_covered_layer_types,
            frozenset({"full_attention"}),
        )
        self.assertIsNotNone(context)
        assert context is not None
        self.assertIs(context.kv_pool, pool)
        self.assertIs(context.attention_workspace, backend.attention_workspace)
        self.assertIs(context.metadata_for_layer(7, "full_attention"), by_layer)
        self.assertIsNone(context.metadata_for_layer(8, "full_attention"))
        self.assertIs(
            context.paged_metadata_for_layer(7, "full_attention"),
            paged_by_layer,
        )
        self.assertIsNone(context.paged_metadata_for_layer(8, "full_attention"))
        self.assertEqual(
            context.covered_decode_layer_types(),
            frozenset({"full_attention"}),
        )

        backend.clear_decode_batch_state()
        self.assertIsNone(backend.current_decode_batch_state)
        self.assertEqual(backend.current_covered_layer_types, frozenset())
        self.assertIsNone(backend.build_current_decode_context())

        typed_state = backend.set_decode_batch_state_by_layer_type(
            metadata_by_layer_type={
                "sliding_attention": by_type,
                "full_attention": by_layer,
            },
            paged_metadata_by_layer_type={
                "sliding_attention": paged_by_type,
                "full_attention": paged_by_layer,
            },
            layer_type_by_layer_id={
                6: "sliding_attention",
                7: "full_attention",
                8: "unknown",
            },
        )
        typed_context = backend.build_current_decode_context(
            layer_id_metadata_only_types=frozenset({"full_attention"}),
        )
        self.assertIs(backend.current_decode_batch_state, typed_state)
        self.assertEqual(
            backend.current_covered_layer_types,
            frozenset({"full_attention", "sliding_attention"}),
        )
        self.assertIs(typed_state.metadata_by_layer_id[6], by_type)
        self.assertIs(typed_state.metadata_by_layer_id[7], by_layer)
        self.assertNotIn(8, typed_state.metadata_by_layer_id)
        self.assertIs(typed_state.paged_metadata_by_layer_id[6], paged_by_type)
        self.assertIs(typed_state.paged_metadata_by_layer_id[7], paged_by_layer)
        self.assertIsNotNone(typed_context)
        assert typed_context is not None
        self.assertIs(
            typed_context.metadata_for_layer(6, "sliding_attention"),
            by_type,
        )
        self.assertIs(
            typed_context.metadata_for_layer(7, "full_attention"),
            by_layer,
        )
        self.assertIsNone(typed_context.metadata_for_layer(8, "unknown"))

    def test_token_pool_decode_backend_builds_layer_plan(self) -> None:
        from types import SimpleNamespace

        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenPoolDecodeBackendState,
        )

        pool = SimpleNamespace(
            layer_specs={0: object(), 1: object(), 2: object()},
            target_layer=lambda layer_id: 1 if int(layer_id) == 2 else int(layer_id),
        )
        backend = TokenPoolDecodeBackendState(
            table=ReqToTokenTable(max_requests=1, max_context_len=8),
            kv_pool=pool,
        )

        plan = backend.build_layer_plan(
            layer_types=(
                "sliding_attention",
                "full_attention",
                "full_attention",
            ),
            model_layer_ids=[0, 1, 2],
            expected_full_attention_owner_layer_ids=[1],
        )

        self.assertEqual(
            plan.layer_type_by_layer_id,
            {
                0: "sliding_attention",
                1: "full_attention",
                2: "full_attention",
            },
        )
        self.assertEqual(plan.full_attention_owner_layer_ids, (1,))
        self.assertEqual(plan.full_attention_layer_ids, (1, 2))
        self.assertEqual(plan.pool_full_attention_layer_ids, (1, 2))
        self.assertTrue(plan.supports_full_attention_decode_metadata)

        unsupported = backend.build_layer_plan(
            layer_types=(
                "sliding_attention",
                "full_attention",
                "full_attention",
            ),
            model_layer_ids=[0, 1, 2],
            expected_full_attention_owner_layer_ids=[2],
        )
        self.assertFalse(unsupported.supports_full_attention_decode_metadata)

    def test_token_pool_decode_backend_backfills_prefill_tail(self) -> None:
        from types import SimpleNamespace

        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenPoolDecodeBackendState,
        )

        class FakePool:
            def __init__(self) -> None:
                self.layer_specs = {0: object()}
                self.writes = []

            def target_layer(self, layer_id):
                return int(layer_id)

            def set_kv(self, layer_id, slots, keys, values) -> None:
                self.writes.append(
                    {
                        "layer_id": int(layer_id),
                        "slots": list(slots),
                        "keys": keys.clone(),
                        "values": values.clone(),
                    }
                )

        class FakeCache:
            def __init__(self, layer) -> None:
                self.layers = [layer]
                self.released = None

            def release_token_pool_covered_sliding_storage(self, layer_types) -> None:
                self.released = set(layer_types)

        layer = SimpleNamespace(
            is_sliding=True,
            keys=torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2),
            values=torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2) + 100,
            _dense_storage_released=False,
        )
        cache = FakeCache(layer)
        pool = FakePool()
        backend = TokenPoolDecodeBackendState(
            table=ReqToTokenTable(max_requests=1, max_context_len=8),
            kv_pool=pool,
        )

        self.assertEqual(backend.available_prefill_tail(cache, 4), 3)
        backend.backfill_prefill_tokens(
            cache,
            torch.tensor([8, 9, 10], dtype=torch.int32),
            2,
            token_slot_ids=[8, 9, 10],
            release_covered=True,
        )
        backend.release_prefill_sliding_storage(cache)

        self.assertEqual(len(pool.writes), 1)
        write = pool.writes[0]
        self.assertEqual(write["layer_id"], 0)
        self.assertEqual(write["slots"], [9, 10])
        self.assertTrue(
            torch.equal(
                write["keys"],
                torch.tensor([[[2.0, 3.0]], [[4.0, 5.0]]]),
            )
        )
        self.assertTrue(
            torch.equal(
                write["values"],
                torch.tensor([[[102.0, 103.0]], [[104.0, 105.0]]]),
            )
        )
        self.assertIsNone(layer.keys)
        self.assertIsNone(layer.values)
        self.assertTrue(layer._dense_storage_released)
        self.assertEqual(cache.released, {"sliding_attention"})

    def test_token_pool_decode_backend_commits_prefill_normal_slots(self) -> None:
        from types import SimpleNamespace

        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenPoolDecodeBackendState,
            TokenSlotAllocator,
        )

        table = ReqToTokenTable(max_requests=1, max_context_len=4)
        allocator = TokenSlotAllocator(capacity=8)
        backend = TokenPoolDecodeBackendState(table=table, allocator=allocator)

        result = backend.commit_prefill_tokens(
            SimpleNamespace(req_id="req"),
            3,
            expected_length=0,
        )

        self.assertEqual(result.req_id, "req")
        self.assertEqual(result.previous_length, 0)
        self.assertEqual(result.new_length, 3)
        self.assertEqual(result.kept_tokens, 3)
        self.assertEqual(result.padded_tokens, 0)
        self.assertFalse(result.backfilled)
        self.assertEqual(result.allocated_token_slots, (0, 1, 2))
        self.assertEqual(table.slots_for("req").tolist(), [0, 1, 2])
        self.assertEqual(backend.request_token_slots["req"], [0, 1, 2])
        self.assertEqual(allocator.allocated_count, 3)

    def test_token_pool_decode_backend_prefill_rollback_frees_normal_slots(self) -> None:
        from types import SimpleNamespace

        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenPoolDecodeBackendState,
            TokenSlotAllocator,
        )

        table = ReqToTokenTable(max_requests=1, max_context_len=4)
        allocator = TokenSlotAllocator(capacity=8)
        backend = TokenPoolDecodeBackendState(table=table, allocator=allocator)
        req_slot = backend.admit_request("req")
        original_append_table_slots = backend.append_table_slots

        def raise_append_failure(*args, **kwargs):
            raise RuntimeError("forced prefill append failure")

        backend.append_table_slots = raise_append_failure  # type: ignore[method-assign]
        try:
            with self.assertRaisesRegex(RuntimeError, "forced prefill append failure"):
                backend.commit_prefill_tokens(
                    SimpleNamespace(req_id="req"),
                    2,
                    expected_length=0,
                )
        finally:
            backend.append_table_slots = original_append_table_slots  # type: ignore[method-assign]

        self.assertEqual(table.length(req_slot), 0)
        self.assertEqual(backend.request_token_slots["req"], [])
        self.assertEqual(allocator.allocated_count, 0)

    def test_token_pool_decode_backend_prefill_backfills_paged_tail(self) -> None:
        from types import SimpleNamespace

        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenKVLayerSpec,
            TokenKVPool,
            TokenPoolBlockTables,
            TokenPoolDecodeBackendState,
        )

        table = ReqToTokenTable(max_requests=1, max_context_len=8)
        block_tables = TokenPoolBlockTables(
            max_requests=1,
            max_context_len=8,
            block_size=4,
        )
        pool = TokenKVPool(
            capacity=8,
            layer_specs=[
                TokenKVLayerSpec(
                    layer_id=0,
                    num_kv_heads=1,
                    head_dim=2,
                    dtype=torch.float32,
                )
            ],
            defer_buffer_allocation=True,
        )
        backend = TokenPoolDecodeBackendState(
            table=table,
            allocator=pool,
            kv_pool=pool,
            block_tables=block_tables,
            block_size=4,
        )
        keys = torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2)
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

        result = backend.commit_prefill_tokens(
            SimpleNamespace(req_id="req"),
            6,
            expected_length=0,
            cache=cache,
            sliding_window=4,
            final_prefill=True,
        )

        self.assertEqual(result.kept_tokens, 3)
        self.assertEqual(result.padded_tokens, 3)
        self.assertTrue(result.backfilled)
        self.assertEqual(result.allocated_token_slots, (3, 4, 5))
        self.assertEqual(
            table.slots_for("req").tolist(),
            [table.padding_token, table.padding_token, table.padding_token, 3, 4, 5],
        )
        self.assertEqual(backend.request_token_slots["req"], [])
        self.assertEqual(backend.page_table_for_request("req"), {0: 0, 1: 1})
        self.assertEqual(block_tables.tensor.tolist(), [[0, 1]])
        self.assertEqual(pool.allocated_count, 8)
        gathered_k, gathered_v = pool.gather_kv(0, [3, 4, 5])
        self.assertTrue(torch.equal(gathered_k, keys[0].permute(1, 0, 2)))
        self.assertTrue(torch.equal(gathered_v, values[0].permute(1, 0, 2)))
        self.assertIsNone(cache.layers[0].keys)
        self.assertIsNone(cache.layers[0].values)

    def test_token_pool_decode_backend_commits_decode_transaction_prefix(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenPoolDecodeBackendState,
            TokenPoolDecodeReservation,
            TokenSlotAllocator,
        )

        table = ReqToTokenTable(max_requests=1, max_context_len=8)
        allocator = TokenSlotAllocator(capacity=8)
        backend = TokenPoolDecodeBackendState(table=table, allocator=allocator)
        req_slot = backend.admit_request("req")
        token_slot_tensor, token_slot_ids = allocator.alloc_slots_with_ids(6)
        backend.append_table_slots(req_slot, token_slot_tensor[:5])
        backend.append_request_token_slots("req", token_slot_ids[:5])
        backend.append_table_slots(req_slot, token_slot_tensor[5:6])
        backend.append_request_token_slot("req", token_slot_ids[5])
        reservation = TokenPoolDecodeReservation(
            req_id="req",
            req_slot=req_slot,
            token_slot=token_slot_ids[5],
            token_slot_tensor=token_slot_tensor[5:6],
            previous_length=5,
        )

        prepared = backend.prepared_decode_batch([reservation])
        result = backend.commit_decode_batch(prepared, attention_window=3)

        self.assertEqual(result.cleared_prefix_slots, tuple(token_slot_ids[:3]))
        self.assertEqual(result.released_prefix_slots, tuple(token_slot_ids[:3]))
        self.assertEqual(result.expired_page_slots, ())
        self.assertEqual(result.invalidated_full_attention_rows, 0)
        self.assertEqual(
            table.slots_for("req").tolist(),
            [-1, -1, -1, token_slot_ids[3], token_slot_ids[4], token_slot_ids[5]],
        )
        self.assertEqual(backend.request_token_slots["req"], token_slot_ids[3:])
        self.assertEqual(allocator.allocated_count, 3)

    def test_token_pool_decode_backend_prepares_decode_reservations(self) -> None:
        from types import SimpleNamespace

        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenPoolDecodeBackendState,
            TokenSlotAllocator,
        )

        table = ReqToTokenTable(max_requests=1, max_context_len=8)
        allocator = TokenSlotAllocator(capacity=8)
        backend = TokenPoolDecodeBackendState(table=table, allocator=allocator)
        req_slot = backend.admit_request("req")
        prefill_tensor, prefill_ids = allocator.alloc_slots_with_ids(2)
        backend.append_table_slots(req_slot, prefill_tensor)
        backend.append_request_token_slots("req", prefill_ids)

        reservations = backend.prepare_decode_reservations(
            [SimpleNamespace(req_id="req")],
            expected_lengths=[2],
        )

        self.assertEqual(len(reservations), 1)
        reservation = reservations[0]
        self.assertEqual(reservation.req_id, "req")
        self.assertEqual(reservation.req_slot, req_slot)
        self.assertEqual(reservation.previous_length, 2)
        self.assertEqual(table.length(req_slot), 3)
        self.assertEqual(
            table.slots_for("req").tolist(),
            prefill_ids + [reservation.token_slot],
        )
        self.assertEqual(
            backend.request_token_slots["req"],
            prefill_ids + [reservation.token_slot],
        )
        self.assertEqual(allocator.allocated_count, 3)

    def test_token_pool_decode_backend_prepare_rolls_back_allocated_slot(self) -> None:
        from types import SimpleNamespace

        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenPoolDecodeBackendState,
            TokenSlotAllocator,
        )

        table = ReqToTokenTable(max_requests=1, max_context_len=8)
        allocator = TokenSlotAllocator(capacity=8)
        backend = TokenPoolDecodeBackendState(table=table, allocator=allocator)
        req_slot = backend.admit_request("req")
        prefill_tensor, prefill_ids = allocator.alloc_slots_with_ids(2)
        backend.append_table_slots(req_slot, prefill_tensor)
        backend.append_request_token_slots("req", prefill_ids)
        original_append_table_slots = backend.append_table_slots

        def raise_after_decode_alloc(*args, **kwargs):
            raise RuntimeError("forced append failure")

        backend.append_table_slots = raise_after_decode_alloc  # type: ignore[method-assign]
        try:
            with self.assertRaisesRegex(RuntimeError, "forced append failure"):
                backend.prepare_decode_reservations(
                    [SimpleNamespace(req_id="req")],
                    expected_lengths=[2],
                )
        finally:
            backend.append_table_slots = original_append_table_slots  # type: ignore[method-assign]

        self.assertEqual(table.length(req_slot), 2)
        self.assertEqual(table.slots_for("req").tolist(), prefill_ids)
        self.assertEqual(backend.request_token_slots["req"], prefill_ids)
        self.assertEqual(allocator.allocated_count, 2)

    def test_token_pool_decode_backend_prepares_decode_batch(self) -> None:
        from types import SimpleNamespace

        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenPoolDecodeBackendState,
            TokenSlotAllocator,
        )

        table = ReqToTokenTable(max_requests=1, max_context_len=8)
        allocator = TokenSlotAllocator(capacity=8)
        backend = TokenPoolDecodeBackendState(table=table, allocator=allocator)
        req_slot = backend.admit_request("req")
        prefill_tensor, prefill_ids = allocator.alloc_slots_with_ids(2)
        backend.append_table_slots(req_slot, prefill_tensor)
        backend.append_request_token_slots("req", prefill_ids)

        prepared = backend.prepare_decode_batch(
            [SimpleNamespace(req_id="req")],
            expected_lengths=[2],
            sliding_window=2,
        )

        self.assertIs(backend.current_decode_batch_state, prepared.state)
        self.assertEqual(len(prepared.reservations), 1)
        reservation = prepared.reservations[0]
        self.assertEqual(reservation.req_id, "req")
        self.assertEqual(table.length(req_slot), 3)
        self.assertEqual(
            backend.request_token_slots["req"],
            prefill_ids + [reservation.token_slot],
        )
        assert prepared.state is not None
        metadata = prepared.state.metadata_by_layer_type["full_attention"]
        self.assertEqual(metadata.logical_seq_lens.tolist(), [3])
        self.assertEqual(metadata.out_cache_loc.tolist(), [reservation.token_slot])

    def test_token_pool_decode_backend_prepare_batch_rolls_back_on_metadata_failure(
        self,
    ) -> None:
        from types import SimpleNamespace

        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenPoolDecodeBackendState,
            TokenSlotAllocator,
        )

        table = ReqToTokenTable(max_requests=1, max_context_len=8)
        allocator = TokenSlotAllocator(capacity=8)
        backend = TokenPoolDecodeBackendState(table=table, allocator=allocator)
        req_slot = backend.admit_request("req")
        prefill_tensor, prefill_ids = allocator.alloc_slots_with_ids(2)
        backend.append_table_slots(req_slot, prefill_tensor)
        backend.append_request_token_slots("req", prefill_ids)

        def raise_metadata_failure(_reservations):
            raise RuntimeError("forced metadata failure")

        with self.assertRaisesRegex(RuntimeError, "forced metadata failure"):
            backend.prepare_decode_batch(
                [SimpleNamespace(req_id="req")],
                expected_lengths=[2],
                sliding_window=2,
                full_attention_metadata_provider=raise_metadata_failure,
            )

        self.assertIsNone(backend.current_decode_batch_state)
        self.assertEqual(table.length(req_slot), 2)
        self.assertEqual(table.slots_for("req").tolist(), prefill_ids)
        self.assertEqual(backend.request_token_slots["req"], prefill_ids)
        self.assertEqual(allocator.allocated_count, 2)

    def test_token_pool_decode_backend_prepares_decode_batch_state(self) -> None:
        from types import SimpleNamespace

        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenPoolDecodeBackendState,
            TokenPoolDecodeReservation,
        )

        table = ReqToTokenTable(max_requests=1, max_context_len=8)
        req_slot = table.allocate("req")
        table.append_slots(req_slot, [1, 2, 3])
        pool = SimpleNamespace(
            layer_specs={0: object()},
            capacity=8,
            device="cpu",
        )
        backend = TokenPoolDecodeBackendState(
            table=table,
            kv_pool=pool,
            token_pool_capacity=8,
        )
        reservation = TokenPoolDecodeReservation(
            req_id="req",
            req_slot=req_slot,
            token_slot=3,
            token_slot_tensor=torch.tensor([3], dtype=torch.int32),
            previous_length=2,
        )

        prepared = backend.prepare_decode_batch_state(
            [reservation],
            sliding_window=2,
            layer_type_by_layer_id={0: "sliding_attention"},
        )
        context = prepared.build_context(
            kv_pool=pool,
            attention_workspace=backend.attention_workspace,
        )

        self.assertIs(backend.current_decode_batch_state, prepared.state)
        self.assertEqual(
            prepared.covered_layer_types,
            frozenset({"sliding_attention"}),
        )
        self.assertIsNotNone(context)
        assert context is not None
        metadata = context.metadata_for_layer(0, "sliding_attention")
        self.assertIsNotNone(metadata)
        self.assertEqual(metadata.seq_lens.tolist(), [2])
        self.assertEqual(metadata.logical_seq_lens.tolist(), [3])
        self.assertEqual(metadata.kv_indices.tolist(), [2, 3])
        self.assertEqual(metadata.out_cache_loc.tolist(), [3])

    def test_token_pool_decode_backend_discards_decode_transaction(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenPoolDecodeBackendState,
            TokenPoolDecodeReservation,
            TokenSlotAllocator,
        )

        table = ReqToTokenTable(max_requests=1, max_context_len=8)
        allocator = TokenSlotAllocator(capacity=8)
        backend = TokenPoolDecodeBackendState(table=table, allocator=allocator)
        req_slot = backend.admit_request("req")
        prefill_tensor, prefill_ids = allocator.alloc_slots_with_ids(2)
        backend.append_table_slots(req_slot, prefill_tensor)
        backend.append_request_token_slots("req", prefill_ids)
        decode_tensor, decode_ids = allocator.alloc_slots_with_ids(1)
        backend.append_table_slots(req_slot, decode_tensor)
        backend.append_request_token_slot("req", decode_ids[0])
        reservation = TokenPoolDecodeReservation(
            req_id="req",
            req_slot=req_slot,
            token_slot=decode_ids[0],
            token_slot_tensor=decode_tensor,
            previous_length=2,
        )

        result = backend.discard_decode_batch([reservation])

        self.assertEqual(result.freed_token_slots, (decode_ids[0],))
        self.assertEqual(result.restored_page_slots, ())
        self.assertEqual(table.length(req_slot), 2)
        self.assertEqual(table.slots_for("req").tolist(), prefill_ids)
        self.assertEqual(backend.request_token_slots["req"], prefill_ids)
        self.assertEqual(allocator.allocated_count, 2)

    def test_token_pool_decode_backend_builds_sliding_metadata_from_block_tables(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenPoolBlockTables,
            TokenPoolDecodeBackendState,
        )

        table = ReqToTokenTable(max_requests=2, max_context_len=16)
        a = table.allocate("a")
        b = table.allocate("b")
        table.append_slots(a, [0, 1, 2, 3, 4, 5])
        table.append_slots(b, [8, 9, 10, 11, 16, 17, 18, 19, 20, 21])

        block_tables = TokenPoolBlockTables(
            max_requests=2,
            max_context_len=16,
            block_size=4,
        )
        block_tables.set_block(a, 0, 0)
        block_tables.set_block(a, 1, 1)
        block_tables.set_block(b, 0, 2)
        block_tables.set_block(b, 1, 4)
        block_tables.set_block(b, 2, 5)
        backend = TokenPoolDecodeBackendState(
            table=table,
            block_tables=block_tables,
            block_size=4,
            page_table_metadata_max_rows=0,
            token_pool_capacity=32,
        )

        metadata, paged = backend.build_sliding_decode_metadata(
            req_slots=[a, b],
            logical_seq_lens=[6, 10],
            out_cache_loc=[5, 21],
            sliding_window=5,
            build_paged_metadata=True,
        )

        self.assertEqual(metadata.req_pool_indices.tolist(), [a, b])
        self.assertEqual(metadata.seq_lens.tolist(), [5, 5])
        self.assertEqual(metadata.logical_seq_lens.tolist(), [6, 10])
        self.assertEqual(
            metadata.kv_indices.tolist(),
            [1, 2, 3, 4, 5, 17, 18, 19, 20, 21],
        )
        self.assertEqual(metadata.out_cache_loc_long.tolist(), [5, 21])
        self.assertIsNotNone(paged)
        self.assertEqual(paged.block_tables.tolist(), [[0, 1], [4, 5]])
        self.assertEqual(paged.block_table_lens.tolist(), [2, 2])
        self.assertEqual(paged.selected_start_positions.tolist(), [1, 5])
        self.assertEqual(paged.slot_mapping.tolist(), [5, 21])
        block_ptr = int(paged.block_tables.data_ptr())
        kv_ptr = int(metadata.kv_indices.data_ptr())

        metadata_again, paged_again = backend.build_sliding_decode_metadata(
            req_slots=[a, b],
            logical_seq_lens=[6, 10],
            out_cache_loc=[5, 21],
            sliding_window=5,
            build_paged_metadata=True,
        )
        self.assertEqual(int(metadata_again.kv_indices.data_ptr()), kv_ptr)
        self.assertEqual(int(paged_again.block_tables.data_ptr()), block_ptr)
        self.assertIs(backend.page_table_tensor, block_tables.tensor)

    def test_token_pool_decode_backend_falls_back_to_dict_page_tables_and_pads(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenPoolDecodeBackendState,
        )

        table = ReqToTokenTable(max_requests=2, max_context_len=16)
        a = table.allocate("a")
        b = table.allocate("b")
        table.append_slots(a, [0, 1, 2, 3, 4, 5])
        table.append_slots(b, [8, 9, 10, 11, 16, 17, 18, 19, 20, 21])
        backend = TokenPoolDecodeBackendState(
            table=table,
            block_size=4,
            page_table_metadata_max_rows=2,
            token_pool_capacity=32,
        )

        self.assertTrue(backend.should_build_sliding_paged_metadata())
        metadata, paged = backend.build_sliding_decode_metadata(
            req_slots=[a, b],
            logical_seq_lens=[2, 10],
            out_cache_loc=[1, 21],
            sliding_window=5,
            build_paged_metadata=True,
            page_tables=[{0: 0}, {1: 4, 2: 5}],
            kv_indices_padding_steps=2,
        )

        self.assertEqual(metadata.seq_lens.tolist(), [2, 5])
        self.assertEqual(metadata.kv_indptr.tolist(), [0, 2, 7])
        self.assertEqual(
            metadata.kv_indices.tolist(),
            [0, 1, 17, 18, 19, 20, 21, 21, 21],
        )
        self.assertEqual(metadata.max_seq_len, 5)
        self.assertIsNotNone(paged)
        self.assertEqual(paged.block_tables.tolist(), [[0, -1], [4, 5]])
        self.assertEqual(paged.block_table_lens.tolist(), [1, 2])
        self.assertEqual(paged.selected_start_positions.tolist(), [0, 5])
        self.assertEqual(paged.slot_mapping.tolist(), [1, 21])

    def test_token_pool_decode_backend_builds_full_attention_metadata_from_workspace(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenPoolDecodeBackendState,
            TokenSlotRowChunks,
        )

        table = ReqToTokenTable(max_requests=2, max_context_len=8)
        backend = TokenPoolDecodeBackendState(
            table=table,
            block_size=2,
            token_pool_capacity=16,
        )
        self.assertIs(
            backend.full_attention_decode_metadata_workspace,
            backend.decode_metadata_workspace.flat_workspace("full_attention"),
        )

        first = backend.build_full_attention_decode_metadata(
            rows=[
                TokenSlotRowChunks(
                    (
                        torch.tensor([0, 1], dtype=torch.int32),
                        torch.tensor([2], dtype=torch.int32),
                    ),
                    trusted=True,
                ),
                TokenSlotRowChunks(
                    (
                        torch.tensor([4], dtype=torch.int32),
                        torch.tensor([5], dtype=torch.int32),
                    ),
                    trusted=True,
                ),
            ],
            req_slots=[0, 1],
            logical_seq_lens=[3, 2],
            out_cache_loc=[2, 5],
            kv_indices_padding_steps=1,
        )
        flat_ptrs = {
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
        self.assertEqual(first.kv_indices.tolist(), [0, 1, 2, 4, 5, 5, 5])
        self.assertEqual(first.kv_indptr.tolist(), [0, 3, 5])
        self.assertEqual(first.max_seq_len, 4)

        second = backend.build_full_attention_decode_metadata(
            rows=[
                TokenSlotRowChunks(
                    (
                        torch.tensor([6, 7], dtype=torch.int32),
                        torch.tensor([8], dtype=torch.int32),
                    ),
                    trusted=True,
                ),
                TokenSlotRowChunks(
                    (
                        torch.tensor([10], dtype=torch.int32),
                        torch.tensor([11], dtype=torch.int32),
                    ),
                    trusted=True,
                ),
            ],
            req_slots=[0, 1],
            logical_seq_lens=[3, 2],
            out_cache_loc=[8, 11],
            kv_indices_padding_steps=1,
        )
        for name, ptr in flat_ptrs.items():
            self.assertEqual(int(getattr(second, name).data_ptr()), ptr)
        self.assertEqual(second.kv_indices.tolist(), [6, 7, 8, 10, 11, 11, 11])
        self.assertEqual(second.out_cache_loc_long.dtype, torch.long)

        paged = backend.build_full_attention_paged_decode_metadata(
            paged_rows=[[0, 1, 4, 5], [8, 9]],
            req_slots=[0, 1],
            logical_seq_lens=[2, 2],
            out_cache_loc=[5, 9],
            kv_indices_padding_steps=2,
        )
        paged_block_ptr = int(paged.block_tables.data_ptr())
        paged_start_ptr = int(paged.selected_start_positions.data_ptr())
        self.assertEqual(paged.seq_lens.tolist(), [4, 2])
        self.assertEqual(paged.logical_seq_lens.tolist(), [2, 2])
        self.assertEqual(paged.block_tables.tolist(), [[0, 2, -1], [4, -1, -1]])
        self.assertEqual(paged.block_table_lens.tolist(), [2, 1])
        self.assertEqual(paged.selected_start_positions.tolist(), [0, 0])
        self.assertEqual(paged.slot_mapping.tolist(), [5, 9])
        self.assertEqual(paged.max_seq_len, 6)

        paged_again = backend.build_full_attention_paged_decode_metadata(
            paged_rows=[[2, 3, 6, 7], [10, 11]],
            req_slots=[0, 1],
            logical_seq_lens=[2, 2],
            out_cache_loc=[7, 11],
            kv_indices_padding_steps=2,
        )
        self.assertEqual(int(paged_again.block_tables.data_ptr()), paged_block_ptr)
        self.assertEqual(
            int(paged_again.selected_start_positions.data_ptr()),
            paged_start_ptr,
        )
        self.assertEqual(paged_again.block_tables.tolist(), [[1, 3, -1], [5, -1, -1]])
        self.assertIs(
            backend.decode_metadata_workspace.paged_workspaces["full_attention_paged"],
            backend.decode_metadata_workspace.paged_workspace("full_attention_paged"),
        )

    def test_decode_backend_prepares_full_attention_batch_and_backfills_kv(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from types import SimpleNamespace

        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenKVLayerSpec,
            TokenKVPool,
            TokenPoolDecodeBackendState,
        )

        class FullAttentionLayer:
            def __init__(self) -> None:
                self.keys = torch.arange(6, dtype=torch.float32).reshape(1, 1, 3, 2)
                self.values = self.keys + 100
                self.cumulative_length = 7
                self.write_calls = []

            def materialized_tokens(self) -> int:
                return int(self.keys.shape[2])

            def write_materialized_readout_to_token_pool(
                self,
                token_kv_pool,
                token_slots,
                *,
                layer_id=None,
                token_slots_long=None,
                token_slot_ids=None,
            ) -> None:
                self.write_calls.append(
                    {
                        "layer_id": int(layer_id),
                        "token_slot_ids": list(token_slot_ids or []),
                    }
                )
                slots = token_slots_long if token_slots_long is not None else token_slots
                token_kv_pool.set_kv(
                    int(layer_id),
                    slots,
                    self.keys[0].permute(1, 0, 2).contiguous(),
                    self.values[0].permute(1, 0, 2).contiguous(),
                )

        table = ReqToTokenTable(max_requests=1, max_context_len=8)
        pool = TokenKVPool(
            capacity=16,
            layer_specs=[
                TokenKVLayerSpec(
                    layer_id=1,
                    num_kv_heads=1,
                    head_dim=2,
                    dtype=torch.float32,
                )
            ],
            dtype=torch.float32,
            device="cpu",
        )
        backend = TokenPoolDecodeBackendState(
            table=table,
            allocator=pool,
            kv_pool=pool,
            block_size=4,
            token_pool_capacity=pool.capacity,
        )
        layer = FullAttentionLayer()
        request = SimpleNamespace(req_id="req")
        reservation = SimpleNamespace(
            req_slot=0,
            token_slot=8,
            token_slot_tensor=torch.tensor([8], dtype=torch.int32),
            full_attention_token_slot=None,
        )

        first = backend.prepare_full_attention_decode_batch(
            requests=[request],
            reservations=[reservation],
            caches_by_req_id={"req": SimpleNamespace(layers=[None, layer])},
            owner_layer_ids=[1],
            kv_indices_padding_steps=1,
            persistent_rows=True,
        )

        self.assertEqual(len(layer.write_calls), 1)
        self.assertEqual(layer.write_calls[0]["token_slot_ids"], [0, 1, 2])
        self.assertEqual(reservation.full_attention_token_slot, 3)
        self.assertEqual(first.out_cache_loc, (3,))
        self.assertEqual(first.metadata.out_cache_loc.tolist(), [3])
        self.assertEqual(first.metadata.kv_indices.tolist(), [0, 1, 2, 3, 3])
        self.assertEqual(first.rebuilt_persistent_rows, 1)
        gathered_k, gathered_v = pool.gather_kv(1, [0, 1, 2])
        self.assertTrue(torch.equal(gathered_k, layer.keys[0].permute(1, 0, 2)))
        self.assertTrue(torch.equal(gathered_v, layer.values[0].permute(1, 0, 2)))

        layer.keys = torch.arange(8, dtype=torch.float32).reshape(1, 1, 4, 2)
        layer.values = layer.keys + 100
        layer.cumulative_length = 8
        reuse_reservation = SimpleNamespace(
            req_slot=0,
            token_slot=9,
            token_slot_tensor=torch.tensor([9], dtype=torch.int32),
            full_attention_token_slot=None,
        )
        second = backend.prepare_full_attention_decode_batch(
            requests=[request],
            reservations=[reuse_reservation],
            caches_by_req_id={"req": SimpleNamespace(layers=[None, layer])},
            owner_layer_ids=[1],
            persistent_rows=True,
        )

        self.assertEqual(len(layer.write_calls), 1)
        self.assertEqual(reuse_reservation.full_attention_token_slot, 4)
        self.assertEqual(second.reused_existing_rows, 1)
        self.assertEqual(second.appended_existing_rows, 1)
        self.assertEqual(second.rebuilt_persistent_rows, 0)
        self.assertEqual(second.metadata.kv_indices.tolist(), [0, 1, 2, 3, 4])

        layer_plan = backend.build_layer_plan(
            layer_types=("sliding_attention", "full_attention"),
            model_layer_ids=[1],
            expected_full_attention_owner_layer_ids=[1],
        )
        failed = backend.prepare_full_attention_decode_metadata(
            requests=[request],
            reservations=[
                SimpleNamespace(
                    req_slot=0,
                    token_slot=10,
                    token_slot_tensor=torch.tensor([10], dtype=torch.int32),
                    full_attention_token_slot=None,
                )
            ],
            caches_by_req_id={"req": SimpleNamespace(layers=[None, object()])},
            layer_plan=layer_plan,
            persistent_rows=True,
        )

        self.assertIsNone(failed)
        self.assertEqual(backend.full_attention_row_records, {})

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

    def test_paged_token_slot_rows_accept_typed_decode_metadata_workspace(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            TokenPoolDecodeMetadataWorkspace,
            build_paged_decode_metadata_from_token_slot_rows,
        )

        workspace = TokenPoolDecodeMetadataWorkspace()
        first = build_paged_decode_metadata_from_token_slot_rows(
            [[0, 1, 4, 5], [8, 9]],
            block_size=2,
            block_table_width=3,
            req_slots=[1, 2],
            logical_seq_lens=[4, 2],
            out_cache_loc=[5, 9],
            token_pool_capacity=16,
            padding_block=-7,
            workspace=workspace,
            workspace_key="full_paged",
        )
        ptrs = {
            name: int(getattr(first, name).data_ptr())
            for name in (
                "req_pool_indices",
                "seq_lens",
                "logical_seq_lens",
                "out_cache_loc",
                "out_cache_loc_long",
                "block_table_lens",
                "selected_start_positions",
                "block_tables",
            )
        }

        second = build_paged_decode_metadata_from_token_slot_rows(
            [[2, 3, 6, 7], torch.tensor([10, 11], dtype=torch.int64)],
            block_size=2,
            block_table_width=3,
            req_slots=[3, 4],
            logical_seq_lens=[4, 2],
            out_cache_loc=[7, 11],
            token_pool_capacity=16,
            padding_block=-7,
            workspace=workspace,
            workspace_key="full_paged",
        )

        for name, ptr in ptrs.items():
            self.assertEqual(int(getattr(second, name).data_ptr()), ptr)
        self.assertEqual(second.req_pool_indices.tolist(), [3, 4])
        self.assertEqual(second.seq_lens.tolist(), [4, 2])
        self.assertEqual(second.logical_seq_lens.tolist(), [4, 2])
        self.assertEqual(second.out_cache_loc.tolist(), [7, 11])
        self.assertEqual(second.out_cache_loc_long.dtype, torch.long)
        self.assertEqual(second.block_table_lens.tolist(), [2, 1])
        self.assertEqual(second.selected_start_positions.tolist(), [0, 0])
        self.assertEqual(second.block_tables.tolist(), [[1, 3, -7], [5, -7, -7]])
        self.assertIs(
            workspace.paged_workspaces["full_paged"],
            workspace.paged_workspace("full_paged"),
        )

        separate = build_paged_decode_metadata_from_token_slot_rows(
            [[12, 13]],
            block_size=2,
            req_slots=[5],
            logical_seq_lens=[2],
            out_cache_loc=[13],
            token_pool_capacity=16,
            workspace=workspace,
            workspace_key="sliding_paged",
        )
        self.assertNotEqual(int(separate.block_tables.data_ptr()), ptrs["block_tables"])
        self.assertIn("sliding_paged", workspace.paged_workspaces)

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

    def test_decode_context_owns_mask_and_covered_layer_policy(self) -> None:
        from types import SimpleNamespace

        from wkvm.runner.gemma_token_pool import (
            TokenPoolDecodeBackendState,
            TokenPoolDecodeContext,
        )

        full_mask = object()
        sliding_mask = object()
        mask = {"full_attention": full_mask, "sliding_attention": sliding_mask}
        context = TokenPoolDecodeContext(
            metadata_by_layer_type={
                "sliding_attention": SimpleNamespace(out_cache_loc=object())
            },
            kv_pool=object(),
        )

        self.assertEqual(
            context.covered_decode_layer_types(),
            frozenset({"sliding_attention"}),
        )
        self.assertEqual(
            TokenPoolDecodeBackendState.covered_decode_layer_types(context),
            frozenset({"sliding_attention"}),
        )
        adjusted = context.attention_mask_for_decode(mask)
        self.assertIsNot(adjusted, mask)
        self.assertIs(adjusted["full_attention"], full_mask)
        self.assertIsNone(adjusted["sliding_attention"])
        self.assertIs(mask["sliding_attention"], sliding_mask)

        explicit = TokenPoolDecodeContext(
            metadata_by_layer_type={
                "sliding_attention": SimpleNamespace(out_cache_loc=object()),
                "full_attention": SimpleNamespace(out_cache_loc=None),
            },
            kv_pool=object(),
            covered_layer_types=frozenset({"full_attention"}),
        )
        self.assertEqual(
            explicit.covered_decode_layer_types(),
            frozenset({"full_attention"}),
        )
        explicit_adjusted = TokenPoolDecodeBackendState.attention_mask_for_decode(
            mask,
            explicit,
        )
        self.assertIsNone(explicit_adjusted["full_attention"])
        self.assertIs(explicit_adjusted["sliding_attention"], sliding_mask)

    def test_attention_binding_owns_current_kv_write(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from types import SimpleNamespace
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
                self.buffer_calls = []
                self.buffers = (object(), object())

            def set_kv(self, layer_idx, out_cache_loc, key_states, value_states):
                self.calls.append(
                    (layer_idx, out_cache_loc, key_states, value_states)
                )

            def get_kv_buffer(self, layer_idx):
                self.buffer_calls.append(layer_idx)
                return self.buffers

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
        self.assertTrue(binding.has_kv_pool)
        self.assertTrue(binding.has_write_location())
        self.assertTrue(binding.should_use_decode_attention(query_seq_len=1))
        self.assertFalse(
            binding.should_use_decode_attention(
                attention_mask_present=True,
                query_seq_len=1,
            )
        )
        self.assertFalse(binding.should_use_decode_attention(query_seq_len=2))
        self.assertIs(binding.kv_buffers_for_attention(), pool.buffers)
        self.assertEqual(pool.buffer_calls, [7])
        flat_metadata, paged_metadata = binding.attention_metadata_for_dispatch()
        self.assertIs(flat_metadata, metadata)
        self.assertIsNone(paged_metadata)
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
        self.assertTrue(fallback_binding.has_write_location())
        paged_like_metadata = SimpleNamespace(
            block_tables=object(),
            out_cache_loc=fallback_metadata.out_cache_loc,
        )
        paged_binding = TokenPoolAttentionBinding(
            layer_idx=7,
            metadata=paged_like_metadata,
            paged_metadata=None,
            kv_pool=pool,
        )
        flat_metadata, paged_metadata = paged_binding.attention_metadata_for_dispatch()
        self.assertIsNone(flat_metadata)
        self.assertIs(paged_metadata, paged_like_metadata)
        null_binding = TokenPoolAttentionBinding(
            layer_idx=None,
            metadata=None,
            paged_metadata=None,
            kv_pool=None,
        )
        self.assertFalse(null_binding.has_kv_pool)
        self.assertFalse(null_binding.has_write_location())
        self.assertFalse(null_binding.should_use_decode_attention(query_seq_len=1))
        self.assertIsNone(null_binding.store_current_kv(key_states, value_states))

    def test_attention_plan_resolves_decode_eligibility_and_write(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from types import SimpleNamespace
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
                self.buffer_calls = []
                self.buffers = (object(), object())

            def set_kv(self, layer_idx, out_cache_loc, key_states, value_states):
                self.calls.append(
                    (layer_idx, out_cache_loc, key_states, value_states)
                )

            def get_kv_buffer(self, layer_idx):
                self.buffer_calls.append(layer_idx)
                return self.buffers

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
        self.assertTrue(plan.decode_attention_enabled())
        self.assertIs(plan.metadata, metadata)
        self.assertIs(plan.kv_pool, pool)
        self.assertIs(plan.kv_buffers_for_attention(), pool.buffers)
        self.assertEqual(pool.buffer_calls, [7])
        flat_metadata, paged_metadata = plan.attention_metadata_for_dispatch()
        self.assertIs(flat_metadata, metadata)
        self.assertIsNone(paged_metadata)
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
        long_only_metadata = DecodeBatchMetadata(
            req_pool_indices=metadata.req_pool_indices,
            seq_lens=metadata.seq_lens,
            logical_seq_lens=metadata.logical_seq_lens,
            out_cache_loc=None,
            out_cache_loc_long=metadata.out_cache_loc_long,
            kv_indptr=metadata.kv_indptr,
            kv_indices=metadata.kv_indices,
        )
        long_only_plan = TokenPoolAttentionPlan.from_binding(
            TokenPoolAttentionBinding(
                layer_idx=7,
                metadata=long_only_metadata,
                paged_metadata=None,
                kv_pool=pool,
            ),
            layer_idx=7,
            query_seq_len=1,
        )
        self.assertTrue(long_only_plan.use_decode_attention)
        paged_like_metadata = SimpleNamespace(
            block_tables=object(),
            out_cache_loc=metadata.out_cache_loc,
        )
        paged_plan = TokenPoolAttentionPlan.from_binding(
            TokenPoolAttentionBinding(
                layer_idx=7,
                metadata=paged_like_metadata,
                paged_metadata=None,
                kv_pool=pool,
            ),
            layer_idx=7,
            query_seq_len=1,
        )
        flat_metadata, paged_metadata = paged_plan.attention_metadata_for_dispatch()
        self.assertIsNone(flat_metadata)
        self.assertIs(paged_metadata, paged_like_metadata)

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

        class BindingWithOwnDecision:
            def __init__(self) -> None:
                self.layer_idx = 7
                self.metadata = metadata
                self.paged_metadata = None
                self.kv_pool = pool
                self.calls = []

            def should_use_decode_attention(self, **kwargs):
                self.calls.append(kwargs)
                return False

        decision_binding = BindingWithOwnDecision()
        decision_plan = TokenPoolAttentionPlan.from_binding(
            decision_binding,
            layer_idx=7,
            query_seq_len=1,
        )
        self.assertFalse(decision_plan.use_decode_attention)
        self.assertEqual(
            decision_binding.calls,
            [{"attention_mask_present": False, "query_seq_len": 1}],
        )

    def test_attention_call_owns_plan_kwargs_and_current_kv_routing(self) -> None:
        from wkvm.runner.gemma_token_pool import build_token_pool_attention_call

        events = []

        class MethodOnlyPlan:
            def attention_kwargs(self):
                events.append("kwargs")
                return {
                    "decode_metadata": "metadata",
                    "paged_decode_metadata": "paged_metadata",
                    "token_kv_pool": "pool",
                    "layer_idx": 7,
                }

            def decode_attention_enabled(self):
                events.append("enabled")
                return True

        plan = MethodOnlyPlan()
        call = build_token_pool_attention_call(token_pool_plan=plan)
        self.assertIs(call.plan, plan)
        self.assertEqual(events, ["kwargs", "enabled"])
        self.assertTrue(call.decode_attention_enabled)
        self.assertEqual(
            call.attention_kwargs,
            {
                "decode_metadata": "metadata",
                "paged_decode_metadata": "paged_metadata",
                "token_kv_pool": "pool",
                "layer_idx": 7,
            },
        )
        self.assertEqual(call.current_key_states("key"), "key")
        self.assertEqual(call.current_value_states("value"), "value")
        self.assertIsNone(call.current_key_states("key", is_kv_shared_layer=True))
        self.assertIsNone(call.current_value_states("value", is_kv_shared_layer=True))
        layer_kv = call.bind_layer_kv(
            "key",
            "value",
            has_past_key_values=True,
        )
        self.assertIs(layer_kv.attention_call.plan, plan)
        self.assertEqual(layer_kv.attention_call.key_states_for_write, "key")
        self.assertEqual(layer_kv.attention_call.value_states_for_write, "value")
        self.assertFalse(layer_kv.should_update_dense_cache)
        call_with_kv = call.with_current_kv("key", "value")
        self.assertEqual(call_with_kv.key_states_for_write, "key")
        self.assertEqual(call_with_kv.value_states_for_write, "value")
        self.assertEqual(
            call_with_kv.backend_decode_kwargs(),
            {
                "decode_metadata": "metadata",
                "paged_decode_metadata": "paged_metadata",
                "token_kv_pool": "pool",
                "layer_idx": 7,
                "token_pool_plan": plan,
                "current_key_states": "key",
                "current_value_states": "value",
            },
        )
        self.assertEqual(call_with_kv.current_kv_for_backend(), ("key", "value"))
        shared_call = call.with_current_kv(
            "key",
            "value",
            is_kv_shared_layer=True,
        )
        self.assertIsNone(shared_call.key_states_for_write)
        self.assertIsNone(shared_call.value_states_for_write)
        shared_layer_kv = call.bind_layer_kv(
            "key",
            "value",
            has_past_key_values=True,
            is_kv_shared_layer=True,
        )
        self.assertIsNone(shared_layer_kv.attention_call.key_states_for_write)
        self.assertIsNone(shared_layer_kv.attention_call.value_states_for_write)
        self.assertFalse(shared_layer_kv.should_update_dense_cache)
        self.assertFalse(
            call.should_update_dense_cache(
                has_past_key_values=True,
                is_kv_shared_layer=False,
            )
        )

        direct_metadata = object()
        direct_pool = object()
        direct_call = build_token_pool_attention_call(
            decode_metadata=direct_metadata,
            token_kv_pool=direct_pool,
            layer_idx=3,
            attention_mask_present=False,
            query_seq_len=1,
        )
        self.assertTrue(direct_call.decode_attention_enabled)
        self.assertIsNone(direct_call.plan)
        self.assertEqual(direct_call.attention_kwargs["layer_idx"], 3)
        direct_context = direct_call.backend_dispatch_context()
        self.assertIs(direct_context.flat_metadata, direct_metadata)
        self.assertIs(direct_context.token_kv_pool, direct_pool)
        self.assertEqual(direct_context.layer_idx, 3)
        self.assertEqual(direct_call.current_kv_for_backend(), (None, None))

        masked_call = build_token_pool_attention_call(
            decode_metadata=object(),
            token_kv_pool=object(),
            layer_idx=3,
            attention_mask_present=True,
            query_seq_len=1,
        )
        self.assertFalse(masked_call.decode_attention_enabled)
        self.assertIsNone(masked_call.current_key_states("key"))
        masked_layer_kv = masked_call.bind_layer_kv(
            "key",
            "value",
            has_past_key_values=True,
        )
        self.assertIsNone(masked_layer_kv.attention_call.key_states_for_write)
        self.assertIsNone(masked_layer_kv.attention_call.value_states_for_write)
        self.assertTrue(masked_layer_kv.should_update_dense_cache)
        self.assertTrue(
            masked_call.should_update_dense_cache(
                has_past_key_values=True,
                is_kv_shared_layer=False,
            )
        )
        self.assertFalse(
            masked_call.should_update_dense_cache(
                has_past_key_values=True,
                is_kv_shared_layer=True,
            )
        )

    def test_attention_plan_owns_attention_workspaces(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            TokenPoolAttentionBinding,
            TokenPoolAttentionPlan,
            build_decode_metadata_from_token_slot_rows,
        )

        class WorkspacePool:
            layer_specs = {7: object()}

            def __init__(self) -> None:
                self.output_calls = []
                self.split_calls = []

            def attention_output_buffer(self, **kwargs):
                self.output_calls.append(kwargs)
                return ("pool_output", kwargs)

            def attention_split_workspace(self, **kwargs):
                self.split_calls.append(kwargs)
                return ("pool_split", kwargs)

        metadata = build_decode_metadata_from_token_slot_rows(
            [[3, 4]],
            out_cache_loc=[4],
        )
        pool = WorkspacePool()
        binding = TokenPoolAttentionBinding(
            layer_idx=7,
            metadata=metadata,
            paged_metadata=None,
            kv_pool=pool,
        )
        plan = TokenPoolAttentionPlan.from_binding(
            binding,
            layer_idx=7,
            query_seq_len=1,
        )

        output = plan.attention_output_buffer(
            batch=2,
            query_heads=4,
            head_dim=8,
            dtype="float32",
            device="cpu",
        )
        split = plan.attention_split_workspace(
            batch=2,
            kv_heads=1,
            max_splits=3,
            block_groups=4,
            head_dim=8,
            device="cpu",
        )

        self.assertEqual(output[0], "pool_output")
        self.assertEqual(split[0], "pool_split")
        self.assertEqual(
            pool.output_calls,
            [
                {
                    "batch": 2,
                    "query_heads": 4,
                    "head_dim": 8,
                    "dtype": "float32",
                    "device": "cpu",
                }
            ],
        )
        self.assertEqual(
            pool.split_calls,
            [
                {
                    "batch": 2,
                    "kv_heads": 1,
                    "max_splits": 3,
                    "block_groups": 4,
                    "head_dim": 8,
                    "device": "cpu",
                }
            ],
        )

        class BindingWithOwnWorkspaces:
            def __init__(self) -> None:
                self.layer_idx = 7
                self.metadata = metadata
                self.paged_metadata = None
                self.kv_pool = pool
                self.output_calls = []
                self.split_calls = []

            def attention_output_buffer(self, **kwargs):
                self.output_calls.append(kwargs)
                return "owned_output"

            def attention_split_workspace(self, **kwargs):
                self.split_calls.append(kwargs)
                return "owned_split"

        owned_binding = BindingWithOwnWorkspaces()
        owned_plan = TokenPoolAttentionPlan.from_binding(
            owned_binding,
            layer_idx=7,
            query_seq_len=1,
        )
        self.assertEqual(
            owned_plan.attention_output_buffer(
                batch=1,
                query_heads=2,
                head_dim=4,
                dtype="float16",
                device="cuda",
            ),
            "owned_output",
        )
        self.assertEqual(
            owned_plan.attention_split_workspace(
                batch=1,
                kv_heads=1,
                max_splits=2,
                block_groups=2,
                head_dim=4,
                device="cuda",
            ),
            "owned_split",
        )
        self.assertEqual(len(pool.output_calls), 1)
        self.assertEqual(len(pool.split_calls), 1)
        self.assertEqual(len(owned_binding.output_calls), 1)
        self.assertEqual(len(owned_binding.split_calls), 1)

    def test_attention_plan_owns_dispatch_context(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from types import SimpleNamespace
        from wkvm.runner.gemma_token_pool import (
            DecodeBatchMetadata,
            TokenPoolAttentionBinding,
            TokenPoolAttentionDispatchContext,
            TokenPoolAttentionPlan,
            TokenPoolTritonDecodePlan,
            build_decode_metadata_from_token_slot_rows,
            build_token_pool_attention_dispatch_context,
        )

        class DispatchPool:
            layer_specs = {7: object()}

            def __init__(self) -> None:
                self.buffer_calls = []
                self.output_calls = []
                self.split_calls = []
                self.set_calls = []
                self.buffers = ("key_buffer", "value_buffer")

            def get_kv_buffer(self, layer_idx):
                self.buffer_calls.append(layer_idx)
                return self.buffers

            def set_kv(self, layer_idx, out_cache_loc, key_states, value_states):
                self.set_calls.append(
                    (layer_idx, out_cache_loc, key_states, value_states)
                )

            def attention_output_buffer(self, **kwargs):
                self.output_calls.append(kwargs)
                return ("output", kwargs)

            def attention_split_workspace(self, **kwargs):
                self.split_calls.append(kwargs)
                return ("split", kwargs)

        base_metadata = build_decode_metadata_from_token_slot_rows(
            [[3, 4, 5]],
            out_cache_loc=[5],
        )
        split_plan = TokenPoolTritonDecodePlan(
            should_split=True,
            split_size=2,
            min_splits=2,
            max_splits=2,
        )
        metadata = DecodeBatchMetadata(
            req_pool_indices=base_metadata.req_pool_indices,
            seq_lens=base_metadata.seq_lens,
            logical_seq_lens=base_metadata.logical_seq_lens,
            out_cache_loc=base_metadata.out_cache_loc,
            kv_indptr=base_metadata.kv_indptr,
            kv_indices=base_metadata.kv_indices,
            out_cache_loc_long=base_metadata.out_cache_loc_long,
            max_seq_len=base_metadata.max_seq_len,
            triton_decode_plan=split_plan,
        )
        pool = DispatchPool()
        plan = TokenPoolAttentionPlan.from_binding(
            TokenPoolAttentionBinding(
                layer_idx=7,
                metadata=metadata,
                paged_metadata=None,
                kv_pool=pool,
            ),
            layer_idx=7,
            query_seq_len=1,
        )

        dispatch = plan.attention_dispatch_context()

        self.assertIs(dispatch.flat_metadata, metadata)
        self.assertIsNone(dispatch.paged_metadata)
        self.assertTrue(dispatch.has_flat_metadata)
        self.assertFalse(dispatch.has_paged_metadata)
        self.assertIs(dispatch.kv_buffers_for_attention(), pool.buffers)
        self.assertEqual(pool.buffer_calls, [7])
        self.assertEqual(
            dispatch.attention_output_buffer(
                batch=2,
                query_heads=4,
                head_dim=8,
                dtype="float32",
                device="cpu",
            )[0],
            "output",
        )
        self.assertEqual(
            dispatch.attention_split_workspace(
                batch=2,
                kv_heads=1,
                max_splits=2,
                block_groups=4,
                head_dim=8,
                device="cpu",
            )[0],
            "split",
        )
        self.assertEqual(
            dispatch.triton_split_plan_for_metadata(metadata),
            (True, 2, 2, 2),
        )
        flat_split_dispatch = dispatch.select_triton_dispatch(
            paged_enabled=False,
            split_enabled=True,
            paged_split_enabled=False,
        )
        self.assertEqual(flat_split_dispatch.kind, "flat_split")
        self.assertFalse(flat_split_dispatch.is_paged)
        self.assertTrue(flat_split_dispatch.is_split)
        self.assertIs(flat_split_dispatch.metadata, metadata)
        self.assertEqual(flat_split_dispatch.split_size, 2)
        self.assertEqual(flat_split_dispatch.min_splits, 2)
        self.assertEqual(flat_split_dispatch.max_splits, 2)

        no_split_metadata = DecodeBatchMetadata(
            req_pool_indices=base_metadata.req_pool_indices,
            seq_lens=base_metadata.seq_lens,
            logical_seq_lens=base_metadata.logical_seq_lens,
            out_cache_loc=base_metadata.out_cache_loc,
            kv_indptr=base_metadata.kv_indptr,
            kv_indices=base_metadata.kv_indices,
            out_cache_loc_long=base_metadata.out_cache_loc_long,
            max_seq_len=base_metadata.max_seq_len,
            triton_decode_plan=TokenPoolTritonDecodePlan(
                should_split=False,
                split_size=2,
                min_splits=2,
                max_splits=None,
            ),
        )
        no_split_dispatch = TokenPoolAttentionDispatchContext(
            layer_idx=7,
            flat_metadata=no_split_metadata,
            paged_metadata=None,
            token_kv_pool=pool,
        ).select_triton_dispatch(
            paged_enabled=False,
            split_enabled=True,
            paged_split_enabled=False,
        )
        self.assertEqual(no_split_dispatch.kind, "flat")
        self.assertFalse(no_split_dispatch.is_split)
        self.assertTrue(no_split_dispatch.split_skipped_by_min_splits)

        reference_metadata, reference_pool, reference_layer_idx = (
            dispatch.reference_decode_inputs()
        )
        self.assertIs(reference_metadata, metadata)
        self.assertIs(reference_pool, pool)
        self.assertEqual(reference_layer_idx, 7)
        self.assertIs(
            dispatch.store_current_kv("key_states", "value_states"),
            metadata.out_cache_loc_long,
        )
        self.assertEqual(
            pool.set_calls,
            [(7, metadata.out_cache_loc_long, "key_states", "value_states")],
        )

        paged_like_metadata = SimpleNamespace(
            block_tables=object(),
            block_table_lens=object(),
            selected_start_positions=object(),
            seq_lens=object(),
            block_size=32,
        )
        legacy_dispatch = build_token_pool_attention_dispatch_context(
            decode_metadata=paged_like_metadata,
            token_kv_pool=pool,
            layer_idx=7,
        )
        self.assertIsNone(legacy_dispatch.flat_metadata)
        self.assertIs(legacy_dispatch.paged_metadata, paged_like_metadata)
        self.assertFalse(legacy_dispatch.has_flat_metadata)
        self.assertTrue(legacy_dispatch.has_paged_metadata)
        self.assertIs(legacy_dispatch.kv_buffers_for_attention(), pool.buffers)
        self.assertEqual(pool.buffer_calls, [7, 7])
        paged_dispatch = legacy_dispatch.select_triton_dispatch(
            paged_enabled=True,
            split_enabled=False,
            paged_split_enabled=False,
        )
        self.assertEqual(paged_dispatch.kind, "paged")
        self.assertTrue(paged_dispatch.is_paged)
        self.assertFalse(paged_dispatch.is_split)
        self.assertIs(paged_dispatch.metadata, paged_like_metadata)
        with self.assertRaisesRegex(RuntimeError, "flat decode metadata"):
            legacy_dispatch.select_triton_dispatch(
                paged_enabled=False,
                split_enabled=False,
                paged_split_enabled=False,
            )
        with self.assertRaisesRegex(RuntimeError, "reference fallback"):
            legacy_dispatch.reference_decode_inputs()

        owned_context = TokenPoolAttentionDispatchContext(
            layer_idx=11,
            flat_metadata=metadata,
            paged_metadata=None,
            token_kv_pool=pool,
        )

        class PlanWithOwnDispatch:
            def attention_dispatch_context(self, **kwargs):
                self.kwargs = kwargs
                return owned_context

        owned_plan = PlanWithOwnDispatch()
        self.assertIs(
            build_token_pool_attention_dispatch_context(
                token_pool_plan=owned_plan,
                decode_metadata=base_metadata,
                token_kv_pool=pool,
                layer_idx=7,
            ),
            owned_context,
        )
        self.assertEqual(
            owned_plan.kwargs["decode_metadata"],
            base_metadata,
        )

    def test_attention_module_owns_triton_dispatch_plan_env_cache(self) -> None:
        import os
        from wkvm.runner.gemma_token_pool_attention import (
            TOKEN_POOL_TRITON_DISPATCH_ENV_NAMES,
            reset_token_pool_triton_dispatch_plan_cache,
            token_pool_triton_dispatch_plan,
        )

        old_env = {
            name: os.environ.get(name)
            for name in TOKEN_POOL_TRITON_DISPATCH_ENV_NAMES
        }
        try:
            for name in TOKEN_POOL_TRITON_DISPATCH_ENV_NAMES:
                os.environ.pop(name, None)
            reset_token_pool_triton_dispatch_plan_cache()
            auto = token_pool_triton_dispatch_plan()
            cached_auto = token_pool_triton_dispatch_plan()
            self.assertIs(auto, cached_auto)
            self.assertTrue(auto.effective_enabled)
            self.assertTrue(auto.auto_default_enabled)
            self.assertFalse(auto.paged_enabled)

            os.environ["WKVM_ENABLE_TOKEN_POOL_TRITON"] = "0"
            forced_off = token_pool_triton_dispatch_plan()
            self.assertIsNot(auto, forced_off)
            self.assertFalse(forced_off.effective_enabled)
            self.assertFalse(forced_off.auto_default_enabled)
            self.assertTrue(forced_off.env_forced_off)

            os.environ["WKVM_ENABLE_TOKEN_POOL_TRITON"] = "1"
            os.environ["WKVM_ENABLE_TOKEN_POOL_PAGED_TRITON"] = "1"
            os.environ["WKVM_TOKEN_POOL_TRITON_SPLIT_KV"] = "1"
            os.environ["WKVM_TOKEN_POOL_TRITON_PAGED_SPLIT_KV"] = "1"
            os.environ["WKVM_TOKEN_POOL_TRITON_INPUT_PRECISION"] = "ieee"
            os.environ["WKVM_TOKEN_POOL_TRITON_DOT_DTYPE"] = "native"
            os.environ["WKVM_TOKEN_POOL_TRITON_STRICT"] = "yes"
            explicit = token_pool_triton_dispatch_plan()
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
            reset_token_pool_triton_dispatch_plan_cache()

    def test_attention_backend_owns_reference_fallback_ordering(self) -> None:
        from types import SimpleNamespace
        from wkvm.runner.gemma_token_pool_attention import (
            TokenPoolAttentionBackend,
            TokenPoolAttentionBackendHooks,
            TokenPoolTritonAttentionBackendHooks,
        )

        stats = {
            "calls": 0,
            "env_enabled_calls": 0,
            "env_disabled_calls": 0,
            "effective_enabled_calls": 0,
            "effective_disabled_calls": 0,
            "auto_enabled_calls": 0,
            "paged_enabled_calls": 0,
            "split_enabled_calls": 0,
            "paged_split_enabled_calls": 0,
        }
        events = []
        kv_timings = []
        triton_attempt_timings = []
        attention_timings = []

        class QueryStates:
            is_cuda = False
            shape = (2, 4, 1, 8)

        class DispatchContext:
            def store_current_kv(self, key_states, value_states):
                events.append(("store", key_states, value_states))
                return ("slot0", "slot1")

            def reference_decode_inputs(self):
                events.append(("reference_inputs",))
                return "metadata", "pool", 7

        query_states = QueryStates()
        attn = object()
        dispatch_context = DispatchContext()
        dispatch_plan = SimpleNamespace(
            env_enabled=True,
            env_forced_off=False,
            env_disabled=False,
            effective_enabled=True,
            auto_default_enabled=True,
            paged_enabled=True,
            split_enabled=True,
            paged_split_enabled=True,
        )
        now_values = iter([10.0, 10.25, 10.5, 10.75, 11.0])

        def reference_decode(actual_attn, actual_query_states, **kwargs):
            events.append(("reference_decode", actual_attn, actual_query_states, kwargs))
            return "reference_output", "weights"

        backend = TokenPoolAttentionBackend(
            stats=stats,
            disabled_shapes=set(),
            hooks=TokenPoolAttentionBackendHooks(
                triton=TokenPoolTritonAttentionBackendHooks(
                    decode_fn=lambda: None,
                    split_decode_fn=lambda: None,
                    paged_decode_fn=lambda: None,
                    paged_split_decode_fn=lambda: None,
                    block_groups=lambda groups, dtype: groups,
                    record_fallback=lambda reason: events.append(("fallback", reason)),
                    is_recoverable_runtime_error=lambda exc: True,
                ),
                reference_decode=reference_decode,
                slot_count=len,
                record_kv_write_timing=lambda **kwargs: kv_timings.append(kwargs),
                record_triton_attempt_timing=triton_attempt_timings.append,
                record_attention_timing=lambda kind, rows, elapsed: (
                    attention_timings.append((kind, rows, elapsed))
                ),
                now=lambda: next(now_values),
            ),
        )

        result = backend.decode(
            attn,
            query_states,
            dispatch_context=dispatch_context,
            dispatch_plan=dispatch_plan,
            current_key_states="key_states",
            current_value_states="value_states",
            timing_enabled=True,
        )

        self.assertEqual(result.output, "reference_output")
        self.assertEqual(result.weights, "weights")
        self.assertEqual(result.kind, "reference")
        self.assertEqual(
            events,
            [
                ("store", "key_states", "value_states"),
                ("reference_inputs",),
                (
                    "reference_decode",
                    attn,
                    query_states,
                    {
                        "decode_metadata": "metadata",
                        "token_kv_pool": "pool",
                        "layer_idx": 7,
                    },
                ),
            ],
        )
        self.assertEqual(
            stats,
            {
                "calls": 1,
                "env_enabled_calls": 1,
                "env_disabled_calls": 0,
                "effective_enabled_calls": 1,
                "effective_disabled_calls": 0,
                "auto_enabled_calls": 1,
                "paged_enabled_calls": 1,
                "split_enabled_calls": 1,
                "paged_split_enabled_calls": 1,
            },
        )
        self.assertEqual(kv_timings, [{"tokens": 2, "elapsed": 0.25}])
        self.assertEqual(triton_attempt_timings, [])
        self.assertEqual(attention_timings, [("reference", 2, 1.0)])

    def test_attention_backend_decode_call_builds_dispatch_context(self) -> None:
        from types import SimpleNamespace
        from wkvm.runner.gemma_token_pool import build_token_pool_attention_call
        from wkvm.runner.gemma_token_pool_attention import (
            TokenPoolAttentionBackend,
            TokenPoolAttentionBackendHooks,
            TokenPoolTritonAttentionBackendHooks,
        )

        stats = {
            "calls": 0,
            "env_enabled_calls": 0,
            "env_disabled_calls": 0,
            "effective_enabled_calls": 0,
            "effective_disabled_calls": 0,
            "auto_enabled_calls": 0,
            "paged_enabled_calls": 0,
            "split_enabled_calls": 0,
            "paged_split_enabled_calls": 0,
        }
        events = []

        class QueryStates:
            is_cuda = False
            shape = (1, 4, 1, 8)

        class DispatchContext:
            def store_current_kv(self, key_states, value_states):
                events.append(("store", key_states, value_states))
                return ("slot",)

            def reference_decode_inputs(self):
                events.append(("reference_inputs",))
                return "context_metadata", "context_pool", 9

        class Plan:
            def attention_kwargs(self):
                return {
                    "decode_metadata": "stale_metadata",
                    "paged_decode_metadata": None,
                    "token_kv_pool": "stale_pool",
                    "layer_idx": 3,
                }

            def decode_attention_enabled(self):
                return True

            def attention_dispatch_context(self, **kwargs):
                events.append(("context", kwargs))
                return DispatchContext()

        def reference_decode(actual_attn, actual_query_states, **kwargs):
            events.append(("reference_decode", actual_attn, actual_query_states, kwargs))
            return "output", None

        attention_call = build_token_pool_attention_call(
            token_pool_plan=Plan(),
        ).with_current_kv("key", "value")
        backend = TokenPoolAttentionBackend(
            stats=stats,
            disabled_shapes=set(),
            hooks=TokenPoolAttentionBackendHooks(
                triton=TokenPoolTritonAttentionBackendHooks(
                    decode_fn=lambda: None,
                    split_decode_fn=lambda: None,
                    paged_decode_fn=lambda: None,
                    paged_split_decode_fn=lambda: None,
                    block_groups=lambda groups, dtype: groups,
                    record_fallback=lambda reason: events.append(("fallback", reason)),
                    is_recoverable_runtime_error=lambda exc: True,
                ),
                reference_decode=reference_decode,
                slot_count=len,
                record_kv_write_timing=lambda **kwargs: None,
                record_triton_attempt_timing=lambda elapsed: None,
                record_attention_timing=lambda kind, rows, elapsed: None,
                now=lambda: 0.0,
            ),
        )

        query_states = QueryStates()
        attn = object()
        result = backend.decode_call(
            attn,
            query_states,
            attention_call=attention_call,
            dispatch_plan=SimpleNamespace(
                env_enabled=False,
                env_forced_off=False,
                env_disabled=False,
                effective_enabled=True,
                auto_default_enabled=True,
                paged_enabled=False,
                split_enabled=False,
                paged_split_enabled=False,
            ),
            timing_enabled=False,
        )

        self.assertEqual(result.output, "output")
        self.assertEqual(
            events,
            [
                (
                    "context",
                    {
                        "decode_metadata": "stale_metadata",
                        "paged_decode_metadata": None,
                        "token_kv_pool": "stale_pool",
                        "layer_idx": 3,
                    },
                ),
                ("store", "key", "value"),
                ("reference_inputs",),
                (
                    "reference_decode",
                    attn,
                    query_states,
                    {
                        "decode_metadata": "context_metadata",
                        "token_kv_pool": "context_pool",
                        "layer_idx": 9,
                    },
                ),
            ],
        )
        self.assertEqual(stats["calls"], 1)
        self.assertEqual(stats["auto_enabled_calls"], 1)
        self.assertEqual(stats["effective_enabled_calls"], 1)

    def test_attention_backend_try_decode_call_skips_disabled_call(self) -> None:
        from wkvm.runner.gemma_token_pool import build_token_pool_attention_call
        from wkvm.runner.gemma_token_pool_attention import (
            TokenPoolAttentionBackend,
            TokenPoolAttentionBackendHooks,
            TokenPoolTritonAttentionBackendHooks,
        )

        stats = {
            "calls": 0,
            "env_enabled_calls": 0,
            "env_disabled_calls": 0,
            "effective_enabled_calls": 0,
            "effective_disabled_calls": 0,
            "auto_enabled_calls": 0,
            "paged_enabled_calls": 0,
            "split_enabled_calls": 0,
            "paged_split_enabled_calls": 0,
        }
        events = []

        class Plan:
            def attention_kwargs(self):
                events.append("kwargs")
                return {
                    "decode_metadata": "metadata",
                    "paged_decode_metadata": None,
                    "token_kv_pool": "pool",
                    "layer_idx": 3,
                }

            def decode_attention_enabled(self):
                events.append("enabled")
                return False

            def attention_dispatch_context(self, **kwargs):
                events.append(("context", kwargs))
                raise AssertionError("disabled calls should not build context")

        backend = TokenPoolAttentionBackend(
            stats=stats,
            disabled_shapes=set(),
            hooks=TokenPoolAttentionBackendHooks(
                triton=TokenPoolTritonAttentionBackendHooks(
                    decode_fn=lambda: None,
                    split_decode_fn=lambda: None,
                    paged_decode_fn=lambda: None,
                    paged_split_decode_fn=lambda: None,
                    block_groups=lambda groups, dtype: groups,
                    record_fallback=lambda reason: events.append(("fallback", reason)),
                    is_recoverable_runtime_error=lambda exc: True,
                ),
                reference_decode=lambda *args, **kwargs: (
                    events.append(("reference", args, kwargs))
                    or ("output", None)
                ),
                slot_count=len,
                record_kv_write_timing=lambda **kwargs: events.append(
                    ("kv_timing", kwargs)
                ),
                record_triton_attempt_timing=lambda elapsed: events.append(
                    ("triton_timing", elapsed)
                ),
                record_attention_timing=lambda kind, rows, elapsed: events.append(
                    ("attention_timing", kind, rows, elapsed)
                ),
                now=lambda: 0.0,
            ),
        )
        attention_call = build_token_pool_attention_call(token_pool_plan=Plan())
        events.clear()

        result = backend.try_decode_call(
            object(),
            object(),
            attention_call=attention_call,
            timing_enabled=True,
        )

        self.assertIsNone(result)
        self.assertEqual(events, [])
        self.assertEqual(stats["calls"], 0)
        self.assertEqual(stats["effective_enabled_calls"], 0)

    def test_attention_backend_factory_owns_default_triton_state(self) -> None:
        from types import SimpleNamespace
        from wkvm.runner.gemma_token_pool_attention import (
            build_token_pool_attention_backend,
            clear_token_pool_triton_disabled_shapes,
            is_recoverable_token_pool_triton_error,
            reset_token_pool_triton_fallback_reasons,
            reset_token_pool_triton_stats_counts,
            token_pool_triton_disabled_shapes,
            token_pool_triton_fallback_reasons,
            token_pool_triton_block_groups,
            token_pool_triton_stats_snapshot,
        )

        class QueryStates:
            is_cuda = True
            shape = (1, 4, 1, 8)
            dtype = "fake_dtype"
            device = "cuda:0"

        class DispatchContext:
            has_flat_metadata = True

            def reference_decode_inputs(self):
                return "metadata", "pool", 3

        query_states = QueryStates()
        attn = SimpleNamespace(num_key_value_groups=2, scaling=1.0)
        dispatch_plan = SimpleNamespace(
            env_enabled=True,
            env_forced_off=False,
            env_disabled=False,
            effective_enabled=True,
            auto_default_enabled=False,
            paged_enabled=False,
            split_enabled=False,
            paged_split_enabled=False,
            input_precision_policy="ieee",
            dot_dtype_policy="native",
            strict=False,
        )
        shape_key = (
            4,
            8,
            2,
            query_states.dtype,
            query_states.device,
            "ieee",
            "native",
        )
        reference_calls = []

        def reference_decode(actual_attn, actual_query_states, **kwargs):
            reference_calls.append((actual_attn, actual_query_states, kwargs))
            return "reference", None

        reset_token_pool_triton_stats_counts()
        reset_token_pool_triton_fallback_reasons()
        clear_token_pool_triton_disabled_shapes()
        try:
            token_pool_triton_disabled_shapes().add(shape_key)
            backend = build_token_pool_attention_backend(
                reference_decode=reference_decode,
                slot_count=len,
                record_kv_write_timing=lambda **kwargs: None,
                record_triton_attempt_timing=lambda elapsed: None,
                record_attention_timing=lambda kind, rows, elapsed: None,
                now=lambda: 0.0,
            )
            result = backend.decode(
                attn,
                query_states,
                dispatch_context=DispatchContext(),
                dispatch_plan=dispatch_plan,
                timing_enabled=False,
            )

            stats = token_pool_triton_stats_snapshot()
            self.assertEqual(result.output, "reference")
            self.assertEqual(stats["calls"], 1)
            self.assertEqual(stats["disabled_shape_skips"], 1)
            self.assertEqual(
                token_pool_triton_fallback_reasons(),
                {"disabled_shape": 1},
            )
            block_groups = token_pool_triton_block_groups(3, object())
            self.assertIsInstance(block_groups, int)
            self.assertGreaterEqual(block_groups, 3)
            self.assertTrue(
                is_recoverable_token_pool_triton_error(
                    RuntimeError("out of resource: shared memory")
                )
            )
            self.assertFalse(
                is_recoverable_token_pool_triton_error(
                    RuntimeError("deterministic validation failure")
                )
            )
            self.assertEqual(
                reference_calls,
                [
                    (
                        attn,
                        query_states,
                        {
                            "decode_metadata": "metadata",
                            "token_kv_pool": "pool",
                            "layer_idx": 3,
                        },
                    )
                ],
            )
        finally:
            clear_token_pool_triton_disabled_shapes()
            reset_token_pool_triton_fallback_reasons()
            reset_token_pool_triton_stats_counts()

    def test_token_pool_triton_stats_report_owns_dispatch_fields(self) -> None:
        import os

        from wkvm.runner.gemma_token_pool_attention import (
            TOKEN_POOL_TRITON_DISPATCH_ENV_NAMES,
            clear_token_pool_triton_disabled_shapes,
            record_token_pool_triton_fallback,
            reset_token_pool_triton_dispatch_plan_cache,
            reset_token_pool_triton_fallback_reasons,
            reset_token_pool_triton_stats_counts,
            token_pool_triton_disabled_shapes,
            token_pool_triton_stats_report,
            token_pool_triton_stats_storage,
        )

        old_env = {
            name: os.environ.get(name)
            for name in TOKEN_POOL_TRITON_DISPATCH_ENV_NAMES
        }
        try:
            for name in TOKEN_POOL_TRITON_DISPATCH_ENV_NAMES:
                os.environ.pop(name, None)
            os.environ["WKVM_ENABLE_TOKEN_POOL_TRITON"] = "1"
            os.environ["WKVM_ENABLE_TOKEN_POOL_SPLIT_TRITON"] = "1"
            os.environ["WKVM_ENABLE_TOKEN_POOL_PAGED_SPLIT_TRITON"] = "1"
            os.environ["WKVM_TOKEN_POOL_TRITON_INPUT_PRECISION"] = "ieee"
            os.environ["WKVM_TOKEN_POOL_TRITON_DOT_DTYPE"] = "native"
            reset_token_pool_triton_dispatch_plan_cache()
            reset_token_pool_triton_stats_counts()
            reset_token_pool_triton_fallback_reasons()
            clear_token_pool_triton_disabled_shapes()

            token_pool_triton_stats_storage()["calls"] = 3
            token_pool_triton_disabled_shapes().add(("shape", 1))
            record_token_pool_triton_fallback("runtime")

            stats = token_pool_triton_stats_report(
                split_plan=(False, 128, 2, None),
            )

            self.assertEqual(stats["calls"], 3)
            self.assertEqual(stats["fallback_reasons"], {"runtime": 1})
            self.assertEqual(stats["disabled_shape_count"], 1)
            self.assertTrue(stats["env_enabled"])
            self.assertFalse(stats["env_disabled"])
            self.assertTrue(stats["split_enabled"])
            self.assertTrue(stats["paged_split_enabled"])
            self.assertEqual(stats["split_size"], 128)
            self.assertEqual(stats["split_min_splits"], 2)
            self.assertEqual(stats["input_precision_policy"], "ieee")
            self.assertEqual(stats["dot_dtype_policy"], "native")
            self.assertTrue(stats["effective_enabled"])
            self.assertFalse(stats["auto_default_enabled"])
        finally:
            for name, value in old_env.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            clear_token_pool_triton_disabled_shapes()
            reset_token_pool_triton_fallback_reasons()
            reset_token_pool_triton_stats_counts()
            reset_token_pool_triton_dispatch_plan_cache()

    def test_token_pool_triton_runtime_reset_clears_module_state(self) -> None:
        from wkvm.runner.gemma_token_pool_attention import (
            clear_token_pool_triton_disabled_shapes,
            record_token_pool_triton_fallback,
            reset_token_pool_triton_fallback_reasons,
            reset_token_pool_triton_runtime_state,
            reset_token_pool_triton_stats_counts,
            token_pool_triton_disabled_shapes,
            token_pool_triton_fallback_reasons,
            token_pool_triton_stats_snapshot,
            token_pool_triton_stats_storage,
        )

        reset_token_pool_triton_stats_counts()
        reset_token_pool_triton_fallback_reasons()
        clear_token_pool_triton_disabled_shapes()
        token_pool_triton_stats_storage()["calls"] = 5
        record_token_pool_triton_fallback("runtime")
        token_pool_triton_disabled_shapes().add(("shape", 1))

        reset_token_pool_triton_runtime_state(clear_disabled_shapes=True)

        self.assertEqual(token_pool_triton_stats_snapshot()["calls"], 0)
        self.assertEqual(token_pool_triton_fallback_reasons(), {})
        self.assertEqual(token_pool_triton_disabled_shapes(), set())

    def test_decode_backend_owns_attention_workspace(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenPoolAttentionWorkspace,
            TokenPoolDecodeBackendState,
            build_decode_metadata_from_token_slot_rows,
        )

        workspace = TokenPoolAttentionWorkspace()
        first = workspace.attention_output_buffer(
            batch=2,
            query_heads=4,
            head_dim=8,
            dtype=torch.float32,
            device="cpu",
        )
        second = workspace.attention_output_buffer(
            batch=2,
            query_heads=4,
            head_dim=8,
            dtype=torch.float32,
            device="cpu",
        )
        other = workspace.attention_output_buffer(
            batch=1,
            query_heads=4,
            head_dim=8,
            dtype=torch.float32,
            device="cpu",
        )
        self.assertIs(first, second)
        self.assertEqual(tuple(first.shape), (2, 1, 4, 8))
        self.assertIsNot(first, other)

        split_first = workspace.attention_split_workspace(
            batch=2,
            kv_heads=1,
            max_splits=3,
            block_groups=4,
            head_dim=8,
            device="cpu",
        )
        split_second = workspace.attention_split_workspace(
            batch=2,
            kv_heads=1,
            max_splits=3,
            block_groups=4,
            head_dim=8,
            device="cpu",
        )
        self.assertIs(split_first, split_second)

        class PoolWithoutScratch:
            layer_specs = {7: object()}
            capacity = 16

        table = ReqToTokenTable(max_requests=1, max_context_len=8)
        backend = TokenPoolDecodeBackendState(
            table=table,
            kv_pool=PoolWithoutScratch(),
            block_size=2,
            token_pool_capacity=16,
        )
        metadata = build_decode_metadata_from_token_slot_rows(
            [[3, 4]],
            out_cache_loc=[4],
        )
        context = backend.build_decode_context(
            metadata_by_layer_type={"full_attention": metadata},
        )
        self.assertIs(context.attention_workspace, backend.attention_workspace)
        binding = context.attention_binding_for_layer(7, "full_attention")
        self.assertIs(binding.attention_workspace, backend.attention_workspace)
        plan = context.attention_plan_for_layer(
            7,
            "full_attention",
            query_seq_len=1,
        )
        planned = plan.attention_output_buffer(
            batch=1,
            query_heads=2,
            head_dim=4,
            dtype=torch.float32,
            device="cpu",
        )
        direct = backend.attention_workspace.attention_output_buffer(
            batch=1,
            query_heads=2,
            head_dim=4,
            dtype=torch.float32,
            device="cpu",
        )
        self.assertIs(planned, direct)

    def test_backend_graph_metadata_copy_handles_paged_metadata(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            build_decode_metadata_from_token_slot_rows,
            build_paged_decode_metadata_from_token_slot_rows,
            TokenPoolDecodeBackendState,
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

        graph_metadata = TokenPoolDecodeBackendState.capture_graph_decode_metadata(
            context,
            clone_tensors=True,
        )
        cloned = graph_metadata.context
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
        updated_context = TokenPoolDecodeContext(
            metadata_by_layer_type={"sliding_attention": updated_flat},
            metadata_by_layer_id={0: updated_flat, 1: updated_flat},
            paged_metadata_by_layer_type={"sliding_attention": updated},
            paged_metadata_by_layer_id={0: updated, 1: updated},
            kv_pool=context.kv_pool,
            covered_layer_types=context.covered_layer_types,
        )
        copy_stats = graph_metadata.copy_from(updated_context)
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
        mismatched_context = TokenPoolDecodeContext(
            metadata_by_layer_type={"sliding_attention": updated_flat},
            metadata_by_layer_id={0: updated_flat, 1: updated_flat},
            paged_metadata_by_layer_type={
                "sliding_attention": mismatched_block_size,
            },
            paged_metadata_by_layer_id={
                0: mismatched_block_size,
                1: mismatched_block_size,
            },
            kv_pool=context.kv_pool,
            covered_layer_types=context.covered_layer_types,
        )
        with self.assertRaisesRegex(ValueError, "block_size changed"):
            graph_metadata.copy_from(mismatched_context)

    def test_graph_metadata_replay_compatibility_reports_shape_mismatch(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            build_decode_metadata_from_token_slot_rows,
            TokenPoolDecodeBackendState,
            TokenPoolDecodeContext,
        )

        kv_pool = object()
        captured_metadata = build_decode_metadata_from_token_slot_rows(
            [[0, 1]],
            out_cache_loc=[1],
        )
        captured_context = TokenPoolDecodeContext(
            metadata_by_layer_type={"sliding_attention": captured_metadata},
            metadata_by_layer_id={0: captured_metadata},
            kv_pool=kv_pool,
            covered_layer_types=frozenset({"sliding_attention"}),
        )
        graph_metadata = TokenPoolDecodeBackendState.capture_graph_decode_metadata(
            captured_context,
            clone_tensors=True,
        )

        compatible_metadata = build_decode_metadata_from_token_slot_rows(
            [[2, 3]],
            out_cache_loc=[3],
        )
        compatible_context = TokenPoolDecodeContext(
            metadata_by_layer_type={"sliding_attention": compatible_metadata},
            metadata_by_layer_id={0: compatible_metadata},
            kv_pool=kv_pool,
            covered_layer_types=frozenset({"sliding_attention"}),
        )
        self.assertIsNone(
            graph_metadata.replay_compatibility_error(compatible_context)
        )

        mismatched_metadata = build_decode_metadata_from_token_slot_rows(
            [[4, 5, 6]],
            out_cache_loc=[6],
        )
        mismatched_context = TokenPoolDecodeContext(
            metadata_by_layer_type={"sliding_attention": mismatched_metadata},
            metadata_by_layer_id={0: mismatched_metadata},
            kv_pool=kv_pool,
            covered_layer_types=frozenset({"sliding_attention"}),
        )
        error = graph_metadata.replay_compatibility_error(mismatched_context)
        self.assertIsNotNone(error)
        self.assertIn("metadata_by_layer_type.sliding_attention.kv_indices", error)
        self.assertIn("metadata_by_layer_id.0.kv_indices", error)

    def test_graph_metadata_copy_compatible_from_validates_before_copy(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            build_decode_metadata_from_token_slot_rows,
            TokenPoolDecodeBackendState,
            TokenPoolDecodeContext,
        )

        kv_pool = object()
        captured_metadata = build_decode_metadata_from_token_slot_rows(
            [[0, 1]],
            out_cache_loc=[1],
        )
        captured_context = TokenPoolDecodeContext(
            metadata_by_layer_type={"sliding_attention": captured_metadata},
            metadata_by_layer_id={0: captured_metadata},
            kv_pool=kv_pool,
            covered_layer_types=frozenset({"sliding_attention"}),
        )
        graph_metadata = TokenPoolDecodeBackendState.capture_graph_decode_metadata(
            captured_context,
            clone_tensors=True,
        )

        compatible_metadata = build_decode_metadata_from_token_slot_rows(
            [[2, 3]],
            out_cache_loc=[3],
        )
        compatible_context = TokenPoolDecodeContext(
            metadata_by_layer_type={"sliding_attention": compatible_metadata},
            metadata_by_layer_id={0: compatible_metadata},
            kv_pool=kv_pool,
            covered_layer_types=frozenset({"sliding_attention"}),
        )
        stats = graph_metadata.copy_compatible_from(compatible_context)
        self.assertGreater(stats["cuda_graph_metadata_tensor_copies"], 0)
        self.assertEqual(
            graph_metadata.context.metadata_by_layer_type[
                "sliding_attention"
            ].kv_indices.tolist(),
            [2, 3],
        )

        mismatched_metadata = build_decode_metadata_from_token_slot_rows(
            [[4, 5, 6]],
            out_cache_loc=[6],
        )
        mismatched_context = TokenPoolDecodeContext(
            metadata_by_layer_type={"sliding_attention": mismatched_metadata},
            metadata_by_layer_id={0: mismatched_metadata},
            kv_pool=kv_pool,
            covered_layer_types=frozenset({"sliding_attention"}),
        )
        with self.assertRaisesRegex(
            ValueError,
            "token-pool cuda graph metadata incompatible: "
            ".*metadata_by_layer_type.sliding_attention.kv_indices",
        ):
            graph_metadata.copy_compatible_from(mismatched_context)
        self.assertEqual(
            graph_metadata.context.metadata_by_layer_type[
                "sliding_attention"
            ].kv_indices.tolist(),
            [2, 3],
        )

    def test_graphed_decode_step_rejects_incompatible_token_pool_metadata(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_runner import (
            DistinctCacheBatchError,
            _GraphedPaddedDecodeStep,
        )
        from wkvm.runner.gemma_token_pool import (
            build_decode_metadata_from_token_slot_rows,
            TokenPoolDecodeBackendState,
            TokenPoolDecodeContext,
        )

        kv_pool = object()
        captured_type_metadata = build_decode_metadata_from_token_slot_rows(
            [[0, 1]],
            out_cache_loc=[1],
        )
        captured_id_metadata = build_decode_metadata_from_token_slot_rows(
            [[2, 3]],
            out_cache_loc=[3],
        )
        captured_context = TokenPoolDecodeContext(
            metadata_by_layer_type={"sliding_attention": captured_type_metadata},
            metadata_by_layer_id={0: captured_id_metadata},
            kv_pool=kv_pool,
            covered_layer_types=frozenset({"sliding_attention"}),
        )
        graph_metadata = TokenPoolDecodeBackendState.capture_graph_decode_metadata(
            captured_context,
            clone_tensors=True,
        )
        step = object.__new__(_GraphedPaddedDecodeStep)
        step._token_pool_metadata = graph_metadata

        compatible_type_metadata = build_decode_metadata_from_token_slot_rows(
            [[8, 9]],
            out_cache_loc=[9],
        )
        incompatible_id_metadata = build_decode_metadata_from_token_slot_rows(
            [[10, 11, 12]],
            out_cache_loc=[12],
        )
        incompatible_context = TokenPoolDecodeContext(
            metadata_by_layer_type={"sliding_attention": compatible_type_metadata},
            metadata_by_layer_id={0: incompatible_id_metadata},
            kv_pool=kv_pool,
            covered_layer_types=frozenset({"sliding_attention"}),
        )

        with self.assertRaisesRegex(
            DistinctCacheBatchError,
            "token-pool cuda graph metadata incompatible: .*metadata_by_layer_id.0.kv_indices",
        ):
            step._copy_token_pool_decode_context(incompatible_context)
        self.assertEqual(
            graph_metadata.context.metadata_by_layer_type[
                "sliding_attention"
            ].kv_indices.tolist(),
            [0, 1],
        )

    def test_graph_metadata_facade_aliases_workspace_metadata_and_skips_copies(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenPoolDecodeBackendState,
            TokenPoolDecodeContext,
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

        buffer = TokenPoolDecodeBackendState.capture_graph_decode_metadata(
            context,
            clone_tensors=False,
        )
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

    def test_graph_metadata_copy_compatible_uses_alias_fastpath(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")

        from unittest.mock import patch

        from wkvm.runner.gemma_token_pool import (
            ReqToTokenTable,
            TokenPoolDecodeBackendState,
            TokenPoolDecodeContext,
            TokenPoolDecodeGraphBuffer,
        )

        table = ReqToTokenTable(max_requests=1, max_context_len=8)
        slot = table.allocate("a")
        table.append_slots(slot, [0, 1])
        flat = table.build_decode_metadata(
            [slot],
            out_cache_loc=[1],
            workspace_key="graph_flat",
        )
        context = TokenPoolDecodeContext(
            metadata_by_layer_type={"sliding_attention": flat},
            metadata_by_layer_id={0: flat, 1: flat},
            kv_pool=object(),
            covered_layer_types=frozenset({"sliding_attention"}),
        )
        graph_metadata = TokenPoolDecodeBackendState.capture_graph_decode_metadata(
            context,
            clone_tensors=False,
        )

        table.req_to_token[slot, :2] = torch.tensor([4, 5], dtype=torch.int32)
        updated_flat = table.build_decode_metadata(
            [slot],
            out_cache_loc=[5],
            workspace_key="graph_flat",
        )
        updated = TokenPoolDecodeContext(
            metadata_by_layer_type={"sliding_attention": updated_flat},
            metadata_by_layer_id={0: updated_flat, 1: updated_flat},
            kv_pool=context.kv_pool,
            covered_layer_types=context.covered_layer_types,
        )

        with patch.object(
            TokenPoolDecodeGraphBuffer,
            "replay_compatibility_error",
            side_effect=AssertionError("slow compatibility path used"),
        ), patch.object(
            TokenPoolDecodeGraphBuffer,
            "_copy_decode_metadata_tensor",
            side_effect=AssertionError("slow metadata copy path used"),
        ):
            stats = graph_metadata.copy_compatible_from(updated)

        self.assertEqual(stats["cuda_graph_metadata_tensor_copies"], 0)
        self.assertGreater(stats["cuda_graph_metadata_tensor_copy_skips"], 0)
        self.assertGreater(
            stats["cuda_graph_metadata_alias_fastpath_metadata_skips"],
            0,
        )
        self.assertEqual(
            graph_metadata.context.metadata_by_layer_type[
                "sliding_attention"
            ].kv_indices.tolist(),
            [4, 5],
        )

    def test_graph_signature_tracker_records_reuse_and_mismatch(self) -> None:
        from wkvm.runner.gemma_token_pool import (
            DecodeBatchMetadata,
            TokenPoolDecodeContext,
            TokenPoolDecodeGraphSignatureTracker,
        )

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

        def context(*, kv_indices: int) -> TokenPoolDecodeContext:
            metadata = DecodeBatchMetadata(
                req_pool_indices=FakeTensorShape((2,)),
                seq_lens=FakeTensorShape((2,)),
                logical_seq_lens=FakeTensorShape((2,)),
                out_cache_loc=FakeTensorShape((2,)),
                kv_indptr=FakeTensorShape((3,)),
                kv_indices=FakeTensorShape((kv_indices,)),
            )
            return TokenPoolDecodeContext(
                metadata_by_layer_type={"sliding_attention": metadata},
                kv_pool=object(),
                metadata_by_layer_id={0: metadata},
                covered_layer_types=frozenset({"sliding_attention"}),
            )

        tracker = TokenPoolDecodeGraphSignatureTracker()
        started = tracker.record(("a", "b"), context(kv_indices=4), started_new=True)
        reused = tracker.record(("a", "b"), context(kv_indices=4), started_new=False)
        mismatched = tracker.record(
            ("a", "b"),
            context(kv_indices=5),
            started_new=False,
        )

        self.assertEqual(started.candidate_batches, 1)
        self.assertEqual(started.static_shape_starts, 1)
        self.assertEqual(reused.static_shape_reuses, 1)
        self.assertEqual(mismatched.shape_mismatches, 1)
        self.assertEqual(
            mismatched.shape_mismatch_reasons,
            {
                "metadata_by_layer_type.sliding_attention.kv_indices": 1,
                "metadata_by_layer_id.0.kv_indices": 1,
            },
        )
        self.assertEqual(tracker.discard_touching({"a"}), 1)
        self.assertFalse(tracker.signatures)

    def test_decode_backend_state_owns_graph_signature_records(self) -> None:
        from wkvm.runner.gemma_token_pool import (
            DecodeBatchMetadata,
            TokenPoolDecodeBackendState,
            TokenPoolDecodeContext,
        )

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

        def context(*, kv_indices: int) -> TokenPoolDecodeContext:
            metadata = DecodeBatchMetadata(
                req_pool_indices=FakeTensorShape((2,)),
                seq_lens=FakeTensorShape((2,)),
                logical_seq_lens=FakeTensorShape((2,)),
                out_cache_loc=FakeTensorShape((2,)),
                kv_indptr=FakeTensorShape((3,)),
                kv_indices=FakeTensorShape((kv_indices,)),
            )
            return TokenPoolDecodeContext(
                metadata_by_layer_type={"sliding_attention": metadata},
                kv_pool=object(),
                metadata_by_layer_id={0: metadata},
                covered_layer_types=frozenset({"sliding_attention"}),
            )

        backend = TokenPoolDecodeBackendState(table=object())
        started = backend.record_graph_decode_signature(
            ("a", "b"),
            context(kv_indices=4),
            started_new=True,
        )
        reused = backend.record_graph_decode_signature(
            ("a", "b"),
            context(kv_indices=4),
            started_new=False,
        )
        mismatched = backend.record_graph_decode_signature(
            ("a", "b"),
            context(kv_indices=5),
            started_new=False,
        )

        self.assertEqual(started.static_shape_starts, 1)
        self.assertEqual(reused.static_shape_reuses, 1)
        self.assertEqual(mismatched.shape_mismatches, 1)
        self.assertIn(("a", "b"), backend.graph_decode_signatures)
        self.assertEqual(
            backend.discard_graph_decode_signatures_touching({"a"}),
            1,
        )
        self.assertFalse(backend.graph_decode_signatures)
        backend.record_graph_decode_signature(
            ("c",),
            context(kv_indices=4),
            started_new=True,
        )
        backend.clear_graph_decode_signatures()
        self.assertFalse(backend.graph_decode_signatures)

    def test_graph_decode_context_graphable_requires_cuda_metadata(self) -> None:
        from wkvm.runner.gemma_token_pool import (
            DecodeBatchMetadata,
            TokenPoolDecodeBackendState,
            TokenPoolDecodeContext,
        )

        class FakeTensor:
            dtype = "torch.int32"
            device = "cuda:0"

            def __init__(self, *, is_cuda: bool) -> None:
                self.is_cuda = is_cuda
                self.shape = (1,)

            def numel(self) -> int:
                return 1

        def context(*, is_cuda: bool, kv_pool=object()) -> TokenPoolDecodeContext:
            tensor = FakeTensor(is_cuda=is_cuda)
            metadata = DecodeBatchMetadata(
                req_pool_indices=tensor,
                seq_lens=tensor,
                logical_seq_lens=tensor,
                out_cache_loc=tensor,
                kv_indptr=tensor,
                kv_indices=tensor,
            )
            return TokenPoolDecodeContext(
                metadata_by_layer_type={"sliding_attention": metadata},
                kv_pool=kv_pool,
                metadata_by_layer_id={0: metadata},
                covered_layer_types=frozenset({"sliding_attention"}),
            )

        self.assertFalse(
            TokenPoolDecodeBackendState.graph_decode_context_is_graphable(None)
        )
        self.assertFalse(
            TokenPoolDecodeBackendState.graph_decode_context_is_graphable(
                context(is_cuda=True, kv_pool=None),
            )
        )
        self.assertFalse(
            TokenPoolDecodeBackendState.graph_decode_context_is_graphable(
                context(is_cuda=False),
            )
        )
        self.assertTrue(
            TokenPoolDecodeBackendState.graph_decode_context_is_graphable(
                context(is_cuda=True),
            )
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
