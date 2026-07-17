import unittest

from wkvm.core.mixed_batch import MixedBatchMetadata, MixedBatchRow


class TestMixedBatchMetadata(unittest.TestCase):
    def test_builds_ragged_decode_and_prefill_rows(self) -> None:
        metadata = MixedBatchMetadata.from_rows(
            [
                MixedBatchRow(
                    req_id="decode",
                    prefix_len=12,
                    q_len=1,
                    prompt_len=12,
                    target_len=13,
                ),
                MixedBatchRow(
                    req_id="prefill",
                    prefix_len=4,
                    q_len=3,
                    prompt_len=10,
                    target_len=10,
                    position_start=4,
                ),
            ]
        )

        self.assertEqual(metadata.request_ids, ("decode", "prefill"))
        self.assertEqual(metadata.q_lens, (1, 3))
        self.assertEqual(metadata.q_start_loc, (0, 1, 4))
        self.assertEqual(metadata.cu_q_lens, metadata.q_start_loc)
        self.assertEqual(metadata.positions, (12, 4, 5, 6))
        self.assertEqual(metadata.request_indices, (0, 1, 1, 1))
        self.assertEqual(metadata.query_positions, (0, 0, 1, 2))
        self.assertEqual(metadata.logits_indices, (0, 3))
        self.assertEqual(metadata.sample_mask, (True, False))
        self.assertEqual(metadata.is_prefilling, (False, True))
        self.assertEqual(metadata.decode_row_indices, (0,))
        self.assertEqual(metadata.prefill_row_indices, (1,))
        self.assertEqual(metadata.row_count, 2)
        self.assertEqual(metadata.token_count, 4)

    def test_continuation_extend_is_prefill_work_not_decode(self) -> None:
        metadata = MixedBatchMetadata.from_rows(
            [
                MixedBatchRow(
                    req_id="extend",
                    prefix_len=20,
                    q_len=4,
                    prompt_len=8,
                    target_len=30,
                )
            ]
        )

        self.assertEqual(metadata.is_prefilling, (False,))
        self.assertEqual(metadata.decode_row_indices, ())
        self.assertEqual(metadata.prefill_row_indices, (0,))
        self.assertEqual(metadata.sample_mask, (False,))

    def test_single_token_prompt_rows_are_prefill_rows(self) -> None:
        metadata = MixedBatchMetadata.from_rows(
            [
                MixedBatchRow(
                    req_id="initial-one",
                    prefix_len=0,
                    q_len=1,
                    prompt_len=1,
                    target_len=1,
                    initial=True,
                ),
                MixedBatchRow(
                    req_id="last-prompt-token",
                    prefix_len=2,
                    q_len=1,
                    prompt_len=3,
                    target_len=3,
                ),
            ]
        )

        self.assertEqual(metadata.decode_row_indices, ())
        self.assertEqual(metadata.prefill_row_indices, (0, 1))
        self.assertEqual(metadata.sample_mask, (True, True))

    def test_flatten_token_rows_checks_ragged_widths(self) -> None:
        self.assertEqual(
            MixedBatchMetadata.flatten_token_rows([[1], [2, 3]], [1, 2]),
            (1, 2, 3),
        )
        with self.assertRaises(ValueError):
            MixedBatchMetadata.flatten_token_rows([[1, 2], [3]], [1, 1])
        with self.assertRaises(ValueError):
            MixedBatchMetadata.flatten_token_rows([[]], [0])

    def test_row_slice_and_last_token_index_are_flattened_offsets(self) -> None:
        metadata = MixedBatchMetadata.from_rows(
            [
                MixedBatchRow("one", 0, 2, 2, 2, initial=True),
                MixedBatchRow("two", 4, 1, 4, 5),
            ]
        )

        self.assertEqual(metadata.row_slice(0), slice(0, 2))
        self.assertEqual(metadata.row_slice(1), slice(2, 3))
        self.assertEqual(metadata.last_token_index(0), 1)
        self.assertEqual(metadata.last_token_index(1), 2)
        with self.assertRaises(IndexError):
            metadata.row_slice(2)
        with self.assertRaises(IndexError):
            metadata.last_token_index(-1)

    def test_rejects_duplicate_ids_and_bad_offsets(self) -> None:
        with self.assertRaises(ValueError):
            MixedBatchMetadata.from_rows(
                [
                    MixedBatchRow("same", 0, 1, 4, 1),
                    MixedBatchRow("same", 0, 1, 4, 1),
                ]
            )
        with self.assertRaises(ValueError):
            MixedBatchMetadata(
                request_ids=("one",),
                q_lens=(1,),
                prefix_lens=(0,),
                seq_lens=(1,),
                q_start_loc=(0, 0),
                positions=(),
                logits_indices=(-1,),
                sample_mask=(True,),
                is_prefilling=(True,),
                initial=(False,),
                request_indices=(),
                query_positions=(),
                decode_row_indices=(),
                prefill_row_indices=(),
            )
        with self.assertRaises(ValueError):
            MixedBatchMetadata(
                request_ids=("one",),
                q_lens=(-1,),
                prefix_lens=(0,),
                seq_lens=(-1,),
                q_start_loc=(0, 0),
                positions=(),
                logits_indices=(-1,),
                sample_mask=(True,),
                is_prefilling=(True,),
                initial=(False,),
                request_indices=(),
                query_positions=(),
                decode_row_indices=(),
                prefill_row_indices=(0,),
            )
        with self.assertRaises(ValueError):
            MixedBatchMetadata(
                request_ids=("one",),
                q_lens=(1,),
                prefix_lens=(-1,),
                seq_lens=(0,),
                q_start_loc=(0, 1),
                positions=(0,),
                logits_indices=(0,),
                sample_mask=(True,),
                is_prefilling=(True,),
                initial=(False,),
                request_indices=(0,),
                query_positions=(0,),
                decode_row_indices=(),
                prefill_row_indices=(0,),
            )
        with self.assertRaises(ValueError):
            MixedBatchMetadata(
                request_ids=("one",),
                q_lens=(1,),
                prefix_lens=(2,),
                seq_lens=(3,),
                q_start_loc=(0, 1),
                positions=(7,),
                logits_indices=(0,),
                sample_mask=(True,),
                is_prefilling=(False,),
                initial=(False,),
                request_indices=(0,),
                query_positions=(0,),
                decode_row_indices=(0,),
                prefill_row_indices=(),
            )


if __name__ == "__main__":
    unittest.main()
