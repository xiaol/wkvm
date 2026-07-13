"""Shared prompt utilities for same-workload Gemma benchmark artifacts."""

from __future__ import annotations

import hashlib
import operator
from typing import Iterable


GENERATED_OUTPUT_FINGERPRINT_SCHEMA = (
    "wkvm.generated_output_token_ids.sha256.v1"
)


class SyntheticBenchTokenResult:
    def __init__(self, input_ids: list[int]) -> None:
        self.input_ids = input_ids


class SyntheticBenchTokenizer:
    """Small deterministic token source for tokenizer-free benchmark prompts."""

    def __init__(
        self,
        *,
        vocab_size: int = 262_144,
        bos_token_id: int | None = 2,
        break_period: int = 31,
    ) -> None:
        vocab_size = int(vocab_size)
        if vocab_size < 16:
            raise ValueError("synthetic vocab size must be >= 16")
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id
        self.break_period = max(2, int(break_period))

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> SyntheticBenchTokenResult:
        text = str(text)
        token_count = max(1, (len(text) + 3) // 4)
        usable = self.vocab_size - 4
        seed = 0x345678
        ids: list[int] = []
        for i in range(token_count):
            ch = ord(text[i % len(text)]) if text else 0
            seed = (seed * 1_103_515_245 + ch + i + 12_345) & 0x7FFF_FFFF
            ids.append(4 + (seed % usable))
        if add_special_tokens and self.bos_token_id is not None:
            ids = [int(self.bos_token_id), *ids]
        return SyntheticBenchTokenResult(ids)

    def decode(self, token_ids: Iterable[int]) -> str:
        return "." if any(int(tid) % self.break_period == 0 for tid in token_ids) else "x"


def prompt_set_fingerprint(
    prompts: Iterable[Iterable[int]],
    *,
    prompt_token_source: str,
) -> dict[str, object]:
    """Return a stable workload fingerprint for a set of prompt token IDs."""

    hasher = hashlib.sha256()
    lengths: list[int] = []
    for prompt in prompts:
        token_ids = [int(token_id) for token_id in prompt]
        lengths.append(len(token_ids))
        hasher.update(len(token_ids).to_bytes(8, "little", signed=False))
        for token_id in token_ids:
            hasher.update(int(token_id).to_bytes(8, "little", signed=True))
    return {
        "schema": "wkvm.prompt_token_ids.sha256.v1",
        "prompt_token_source": str(prompt_token_source),
        "prompt_count": len(lengths),
        "prompt_total_tokens": sum(lengths),
        "prompt_lengths": lengths,
        "prompt_token_ids_sha256": hasher.hexdigest(),
    }


def prompt_fingerprint_row_fields(
    fingerprint: dict[str, object],
) -> dict[str, object]:
    return {
        "prompt_fingerprint": dict(fingerprint),
        "prompt_token_source": fingerprint.get("prompt_token_source"),
        "prompt_count": fingerprint.get("prompt_count"),
        "prompt_total_tokens": fingerprint.get("prompt_total_tokens"),
        "prompt_lengths": fingerprint.get("prompt_lengths"),
        "prompt_token_ids_sha256": fingerprint.get("prompt_token_ids_sha256"),
    }


def generated_output_fingerprint(
    request_outputs: Iterable[tuple[str, Iterable[int]]],
) -> dict[str, object]:
    """Return a stable fingerprint of request IDs and generated token IDs."""

    normalized: list[tuple[str, list[int]]] = []
    for request_id, output_token_ids in request_outputs:
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("generated-output request IDs must be non-empty strings")
        token_ids = []
        for token_id in output_token_ids:
            if isinstance(token_id, bool):
                raise ValueError("generated-output token IDs must be integers, not bools")
            try:
                normalized_token_id = operator.index(token_id)
            except TypeError as exc:
                raise ValueError(
                    "generated-output token IDs must be integers"
                ) from exc
            token_ids.append(normalized_token_id)
        normalized.append((request_id, token_ids))

    normalized.sort(key=lambda item: item[0].encode("utf-8"))
    request_ids = [request_id for request_id, _ in normalized]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("generated-output request IDs must be unique")

    hasher = hashlib.sha256()
    hasher.update(GENERATED_OUTPUT_FINGERPRINT_SCHEMA.encode("ascii") + b"\0")
    hasher.update(len(normalized).to_bytes(8, "little", signed=False))
    output_token_counts: list[int] = []
    for request_id, token_ids in normalized:
        request_id_bytes = request_id.encode("utf-8")
        hasher.update(len(request_id_bytes).to_bytes(8, "little", signed=False))
        hasher.update(request_id_bytes)
        hasher.update(len(token_ids).to_bytes(8, "little", signed=False))
        output_token_counts.append(len(token_ids))
        for token_id in token_ids:
            try:
                encoded_token_id = token_id.to_bytes(8, "little", signed=True)
            except OverflowError as exc:
                raise ValueError(
                    "generated-output token IDs must fit signed 64-bit integers"
                ) from exc
            hasher.update(encoded_token_id)

    return {
        "schema": GENERATED_OUTPUT_FINGERPRINT_SCHEMA,
        "request_count": len(normalized),
        "output_token_count": sum(output_token_counts),
        "request_ids": request_ids,
        "output_token_counts": output_token_counts,
        "request_output_token_ids_sha256": hasher.hexdigest(),
    }


def generated_output_fingerprint_row_fields(
    fingerprint: dict[str, object],
) -> dict[str, object]:
    return {
        "generated_output_fingerprint": dict(fingerprint),
        "generated_output_fingerprint_schema": fingerprint.get("schema"),
        "generated_output_request_count": fingerprint.get("request_count"),
        "generated_output_token_count": fingerprint.get("output_token_count"),
        "generated_output_request_ids": fingerprint.get("request_ids"),
        "generated_output_token_counts": fingerprint.get("output_token_counts"),
        "request_output_token_ids_sha256": fingerprint.get(
            "request_output_token_ids_sha256"
        ),
    }
