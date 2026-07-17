"""SGLang custom logit processor for shared teacher-forced traces."""

from __future__ import annotations

from typing import Any

import torch

from sglang.srt.sampling.custom_logit_processor import CustomLogitProcessor


FORCED_TOKEN_IDS_KEY = "wkvm_teacher_forced_token_ids"


class SharedHistoryLogitsProcessor(CustomLogitProcessor):
    """Select request-local trace tokens with one target write per row."""

    def __call__(
        self,
        logits: torch.Tensor,
        custom_param_list: list[dict[str, Any] | None] | None = None,
    ) -> torch.Tensor:
        if not custom_param_list:
            return logits
        target_value = torch.finfo(logits.dtype).max
        for row_index, params in enumerate(custom_param_list):
            if not params:
                continue
            token_ids = params.get(FORCED_TOKEN_IDS_KEY)
            request = params.get("__req__")
            if not isinstance(token_ids, list) or not token_ids:
                raise ValueError(f"{FORCED_TOKEN_IDS_KEY} must be a non-empty list")
            if any(
                isinstance(token, bool) or not isinstance(token, int)
                for token in token_ids
            ):
                raise ValueError(f"{FORCED_TOKEN_IDS_KEY} must contain only integers")
            if request is None:
                raise RuntimeError("SGLang did not attach request state")
            output_index = len(request.output_ids)
            if output_index >= len(token_ids):
                raise RuntimeError("teacher-forced sequence was exhausted")
            logits[row_index, int(token_ids[output_index])] = target_value
        return logits


if __name__ == "__main__":
    print(SharedHistoryLogitsProcessor.to_str())
