"""Request lifecycle.

The single scheduling invariant (stolen from vLLM v1, docs/ANGLE.md §2):
one integer per request — ``num_computed_tokens`` — converging to
``num_tokens`` under a global token budget. Prefill, chunked prefill, decode,
and checkpoint-resume are all the same operation: "schedule some of the gap".
"""

from __future__ import annotations

import enum
import itertools
from dataclasses import dataclass, field


class RequestStatus(enum.Enum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()  # slot released; state swapped out or recomputable
    FINISHED_STOPPED = enum.auto()  # stop token / stop condition
    FINISHED_LENGTH = enum.auto()  # max_new_tokens reached
    FINISHED_ABORTED = enum.auto()

    @property
    def is_finished(self) -> bool:
        return self in (
            RequestStatus.FINISHED_STOPPED,
            RequestStatus.FINISHED_LENGTH,
            RequestStatus.FINISHED_ABORTED,
        )


_req_counter = itertools.count()


@dataclass
class Request:
    """Engine-internal request record. Token ids only — no strings in core."""

    prompt_token_ids: list[int]
    max_new_tokens: int
    req_id: str = field(default_factory=lambda: f"req-{next(_req_counter)}")

    status: RequestStatus = RequestStatus.WAITING
    # Tokens whose state contribution has been computed (prompt + sampled).
    num_computed_tokens: int = 0
    output_token_ids: list[int] = field(default_factory=list)
    # Family name -> slot id, while RUNNING. Owned by the scheduler/arena.
    slots: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.prompt_token_ids:
            raise ValueError("empty prompt")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        """Current convergence target: every known token, prompt + output.

        The final token's state need not be computed to sample from it, so a
        request is schedulable while ``num_computed_tokens < num_tokens``; the
        gap is 1 exactly when the request is in steady-state decode.
        """
        return self.num_prompt_tokens + len(self.output_token_ids)

    @property
    def num_scheduled_gap(self) -> int:
        return self.num_tokens - self.num_computed_tokens

    @property
    def in_prefill(self) -> bool:
        return self.num_computed_tokens < self.num_prompt_tokens - 1

    def reached_length_limit(self) -> bool:
        return len(self.output_token_ids) >= self.max_new_tokens
