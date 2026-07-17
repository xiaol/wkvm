"""vLLM logits processor for shared teacher-forced benchmark traces."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import torch

from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor
from vllm.v1.sample.logits_processor.builtin import process_dict_updates

if TYPE_CHECKING:
    from vllm.config import VllmConfig


FORCED_TOKEN_IDS_KEY = "wkvm_teacher_forced_token_ids"


def parse_forced_token_ids(value: object) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{FORCED_TOKEN_IDS_KEY} string must contain a JSON array"
            ) from exc
    if not isinstance(value, list) or not value:
        raise ValueError(f"{FORCED_TOKEN_IDS_KEY} must be a non-empty list")
    if any(isinstance(token, bool) or not isinstance(token, int) for token in value):
        raise ValueError(f"{FORCED_TOKEN_IDS_KEY} must contain only integers")
    return [int(token) for token in value]


class SharedHistoryLogitsProcessor(LogitsProcessor):
    """Select request-local trace tokens with one batched logit scatter."""

    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        parse_forced_token_ids(
            params.extra_args and params.extra_args.get(FORCED_TOKEN_IDS_KEY)
        )

    def __init__(
        self,
        vllm_config: "VllmConfig",
        device: torch.device,
        is_pin_memory: bool,
    ) -> None:
        del vllm_config, device, is_pin_memory
        self.requests: dict[int, tuple[list[int], list[int]]] = {}

    def is_argmax_invariant(self) -> bool:
        return False

    @classmethod
    def _new_request_state(
        cls,
        params: SamplingParams,
        prompt_token_ids: list[int] | None,
        output_token_ids: list[int],
    ) -> tuple[list[int], list[int]] | None:
        del prompt_token_ids
        token_ids = parse_forced_token_ids(
            params.extra_args and params.extra_args.get(FORCED_TOKEN_IDS_KEY)
        )
        if token_ids is None:
            return None
        return token_ids, output_token_ids

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        process_dict_updates(
            self.requests,
            batch_update,
            self._new_request_state,
        )

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.requests:
            return logits
        row_indices: list[int] = []
        target_tokens: list[int] = []
        for row_index, (token_ids, output_ids) in self.requests.items():
            output_index = len(output_ids)
            if output_index >= len(token_ids):
                raise RuntimeError("teacher-forced sequence was exhausted")
            row_indices.append(row_index)
            target_tokens.append(token_ids[output_index])
        rows = torch.tensor(row_indices, dtype=torch.long, device=logits.device)
        targets = torch.tensor(target_tokens, dtype=torch.long, device=logits.device)
        logits[rows, targets] = float("inf")
        return logits
