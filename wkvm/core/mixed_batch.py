"""Dependency-free metadata for a mixed prefill/decode model call."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class MixedBatchRow:
    """One scheduler row in a ragged model step."""

    req_id: str
    prefix_len: int
    q_len: int
    prompt_len: int
    target_len: int
    position_start: int | None = None
    initial: bool = False

    def __post_init__(self) -> None:
        if not self.req_id:
            raise ValueError("mixed-batch request ids must be non-empty")
        for name in ("prefix_len", "q_len", "prompt_len", "target_len"):
            if int(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.q_len < 1:
            raise ValueError("q_len must be >= 1")
        if self.target_len < self.prefix_len + self.q_len:
            raise ValueError("target_len must cover the scheduled sequence")
        if self.target_len < self.prompt_len:
            raise ValueError("target_len must cover the prompt")
        if self.prompt_len < 1:
            raise ValueError("prompt_len must be >= 1")
        if self.position_start is not None and self.position_start < 0:
            raise ValueError("position_start must be >= 0 or None")
        if self.position_start is not None and int(self.position_start) != self.prefix_len:
            raise ValueError("position_start must equal prefix_len")
        if self.is_prefilling and self.prefix_len + self.q_len > self.prompt_len:
            raise ValueError("prefill rows must not cross the prompt boundary")
        if self.initial and self.prefix_len != 0:
            raise ValueError("initial rows must start at prefix position zero")

    @property
    def position(self) -> int:
        return self.prefix_len if self.position_start is None else int(self.position_start)

    @property
    def is_prefilling(self) -> bool:
        return self.prefix_len < self.prompt_len

    @property
    def is_decode(self) -> bool:
        return not self.is_prefilling and self.q_len == 1

    @property
    def closes_gap(self) -> bool:
        return self.prefix_len + self.q_len == self.target_len


@dataclass(frozen=True)
class MixedBatchMetadata:
    """Validated ragged offsets and row classifications for one model call.

    ``input_ids`` and cache objects deliberately stay outside this core
    structure. The runner can attach device tensors without making the
    scheduler depend on torch or a particular attention backend.
    """

    request_ids: tuple[str, ...]
    q_lens: tuple[int, ...]
    prefix_lens: tuple[int, ...]
    seq_lens: tuple[int, ...]
    q_start_loc: tuple[int, ...]
    positions: tuple[int, ...]
    logits_indices: tuple[int, ...]
    sample_mask: tuple[bool, ...]
    is_prefilling: tuple[bool, ...]
    initial: tuple[bool, ...]
    request_indices: tuple[int, ...]
    query_positions: tuple[int, ...]
    decode_row_indices: tuple[int, ...]
    prefill_row_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        row_count = len(self.request_ids)
        if row_count < 1:
            raise ValueError("mixed-batch metadata requires at least one request")
        if any(not request_id for request_id in self.request_ids):
            raise ValueError("mixed-batch request ids must be non-empty")
        if len(set(self.request_ids)) != row_count:
            raise ValueError("mixed-batch request ids must be unique")
        if any(not isinstance(request_id, str) for request_id in self.request_ids):
            raise ValueError("mixed-batch request ids must be strings")
        if any(int(q_len) < 1 for q_len in self.q_lens):
            raise ValueError("q_lens must be >= 1")
        if any(int(prefix_len) < 0 for prefix_len in self.prefix_lens):
            raise ValueError("prefix_lens must be >= 0")
        if any(int(seq_len) < 0 for seq_len in self.seq_lens):
            raise ValueError("seq_lens must be >= 0")
        if any(
            int(seq_len) < int(prefix_len)
            for seq_len, prefix_len in zip(self.seq_lens, self.prefix_lens)
        ):
            raise ValueError("seq_lens must be >= prefix_lens")
        fields = (
            self.q_lens,
            self.prefix_lens,
            self.seq_lens,
            self.logits_indices,
            self.sample_mask,
            self.is_prefilling,
            self.initial,
        )
        if any(len(field) != row_count for field in fields):
            raise ValueError("mixed-batch row fields must have one value per request")
        if len(self.decode_row_indices) + len(self.prefill_row_indices) != row_count:
            raise ValueError("decode/prefill row partitions must cover every request")
        if len(self.q_start_loc) != row_count + 1:
            raise ValueError("q_start_loc must contain row_count + 1 offsets")
        if self.q_start_loc[0] != 0:
            raise ValueError("q_start_loc must start at zero")
        if any(
            int(offset) < 0
            for offset in self.q_start_loc
        ):
            raise ValueError("q_start_loc must be non-negative")
        if any(
            left > right
            for left, right in zip(self.q_start_loc, self.q_start_loc[1:])
        ):
            raise ValueError("q_start_loc must be monotonic")
        if self.q_start_loc[-1] != len(self.positions):
            raise ValueError("q_start_loc does not cover positions")
        if len(self.request_indices) != len(self.positions):
            raise ValueError("request_indices must cover every flattened token")
        if len(self.query_positions) != len(self.positions):
            raise ValueError("query_positions must cover every flattened token")
        if any(
            not isinstance(index, int) or isinstance(index, bool)
            for index in self.request_indices
        ):
            raise ValueError("request_indices must contain integer row indices")
        if any(
            not isinstance(position, int) or isinstance(position, bool)
            for position in self.query_positions
        ):
            raise ValueError("query_positions must contain integer offsets")
        if any(not isinstance(value, bool) for value in self.sample_mask):
            raise ValueError("sample_mask values must be bool")
        if any(not isinstance(value, bool) for value in self.is_prefilling):
            raise ValueError("is_prefilling values must be bool")
        if any(not isinstance(value, bool) for value in self.initial):
            raise ValueError("initial values must be bool")
        if any(index < 0 or index >= row_count for index in self.decode_row_indices):
            raise ValueError("decode row index is out of range")
        if any(index < 0 or index >= row_count for index in self.prefill_row_indices):
            raise ValueError("prefill row index is out of range")
        if len(set(self.decode_row_indices)) != len(self.decode_row_indices):
            raise ValueError("decode row indices must be unique")
        if len(set(self.prefill_row_indices)) != len(self.prefill_row_indices):
            raise ValueError("prefill row indices must be unique")
        if set(self.decode_row_indices).intersection(self.prefill_row_indices):
            raise ValueError("decode and prefill row partitions must be disjoint")
        if set(self.decode_row_indices).union(self.prefill_row_indices) != set(
            range(row_count)
        ):
            raise ValueError("decode/prefill row partitions must cover every row")
        expected_decode = tuple(
            index
            for index, (prefilling, q_len) in enumerate(
                zip(self.is_prefilling, self.q_lens)
            )
            if not prefilling and q_len == 1
        )
        expected_prefill = tuple(
            index
            for index, (prefilling, q_len) in enumerate(
                zip(self.is_prefilling, self.q_lens)
            )
            if prefilling or q_len > 1
        )
        if self.decode_row_indices != expected_decode:
            raise ValueError("decode_row_indices does not match row metadata")
        if self.prefill_row_indices != expected_prefill:
            raise ValueError("prefill_row_indices does not match row metadata")
        for index, q_len in enumerate(self.q_lens):
            start = self.q_start_loc[index]
            end = self.q_start_loc[index + 1]
            if end - start != q_len:
                raise ValueError("q_start_loc widths do not match q_lens")
            if self.seq_lens[index] != self.prefix_lens[index] + q_len:
                raise ValueError("seq_lens must equal prefix_lens + q_lens")
            if self.logits_indices[index] != end - 1:
                raise ValueError("logits_indices must select each row's last token")
            if self.initial[index] and self.prefix_lens[index] != 0:
                raise ValueError("initial rows must start at prefix position zero")
            expected_request_indices = (index,) * q_len
            if self.request_indices[start:end] != expected_request_indices:
                raise ValueError("request_indices do not match q_lens")
            expected_query_positions = tuple(range(q_len))
            if self.query_positions[start:end] != expected_query_positions:
                raise ValueError("query_positions do not reset at each row")
            expected_positions = tuple(
                range(self.prefix_lens[index], self.prefix_lens[index] + q_len)
            )
            if self.positions[start:end] != expected_positions:
                raise ValueError(
                    "positions must be contiguous absolute positions from prefix_lens"
                )

    @classmethod
    def from_rows(cls, rows: Iterable[MixedBatchRow]) -> "MixedBatchMetadata":
        row_list = tuple(rows)
        if not row_list:
            raise ValueError("mixed-batch metadata requires at least one row")

        request_ids = tuple(row.req_id for row in row_list)
        q_lens = tuple(int(row.q_len) for row in row_list)
        prefix_lens = tuple(int(row.prefix_len) for row in row_list)
        seq_lens = tuple(
            int(row.prefix_len) + int(row.q_len) for row in row_list
        )
        q_start_values = [0]
        for q_len in q_lens:
            q_start_values.append(q_start_values[-1] + q_len)
        q_start_loc = tuple(q_start_values)
        positions = tuple(
            position
            for row in row_list
            for position in range(row.position, row.position + row.q_len)
        )
        logits_indices = tuple(offset - 1 for offset in q_start_loc[1:])
        sample_mask = tuple(row.closes_gap for row in row_list)
        is_prefilling = tuple(row.is_prefilling for row in row_list)
        initial = tuple(bool(row.initial) for row in row_list)
        request_indices = tuple(
            index
            for index, row in enumerate(row_list)
            for _ in range(row.q_len)
        )
        query_positions = tuple(
            query_position
            for row in row_list
            for query_position in range(row.q_len)
        )
        decode_row_indices = tuple(
            index for index, row in enumerate(row_list) if row.is_decode
        )
        prefill_row_indices = tuple(
            index
            for index, row in enumerate(row_list)
            if row.is_prefilling or row.q_len > 1
        )
        return cls(
            request_ids=request_ids,
            q_lens=q_lens,
            prefix_lens=prefix_lens,
            seq_lens=seq_lens,
            q_start_loc=q_start_loc,
            positions=positions,
            logits_indices=logits_indices,
            sample_mask=sample_mask,
            is_prefilling=is_prefilling,
            initial=initial,
            request_indices=request_indices,
            query_positions=query_positions,
            decode_row_indices=decode_row_indices,
            prefill_row_indices=prefill_row_indices,
        )

    @property
    def row_count(self) -> int:
        return len(self.request_ids)

    @property
    def token_count(self) -> int:
        return len(self.positions)

    @property
    def cu_q_lens(self) -> tuple[int, ...]:
        return self.q_start_loc

    @staticmethod
    def flatten_token_rows(
        token_rows: Sequence[Sequence[int]],
        q_lens: Sequence[int],
    ) -> tuple[int, ...]:
        if len(token_rows) != len(q_lens):
            raise ValueError("token row count must match q_lens")
        flattened: list[int] = []
        for tokens, q_len in zip(token_rows, q_lens):
            if int(q_len) < 1:
                raise ValueError("q_lens must be >= 1")
            if len(tokens) != int(q_len):
                raise ValueError("token row width must match q_len")
            flattened.extend(int(token) for token in tokens)
        return tuple(flattened)

    def row_slice(self, row_index: int) -> slice:
        """Return the flattened-token slice for one scheduler row."""

        row_index = int(row_index)
        if row_index < 0 or row_index >= self.row_count:
            raise IndexError("mixed-batch row index is out of range")
        return slice(self.q_start_loc[row_index], self.q_start_loc[row_index + 1])

    def last_token_index(self, row_index: int) -> int:
        """Return the flattened index used to gather that row's logits."""

        row_index = int(row_index)
        if row_index < 0 or row_index >= self.row_count:
            raise IndexError("mixed-batch row index is out of range")
        return self.logits_indices[row_index]

    def as_dict(self) -> dict[str, object]:
        return {
            "request_ids": self.request_ids,
            "q_lens": self.q_lens,
            "prefix_lens": self.prefix_lens,
            "seq_lens": self.seq_lens,
            "q_start_loc": self.q_start_loc,
            "positions": self.positions,
            "logits_indices": self.logits_indices,
            "sample_mask": self.sample_mask,
            "is_prefilling": self.is_prefilling,
            "initial": self.initial,
            "request_indices": self.request_indices,
            "query_positions": self.query_positions,
            "decode_row_indices": self.decode_row_indices,
            "prefill_row_indices": self.prefill_row_indices,
            "row_count": self.row_count,
            "token_count": self.token_count,
        }
