"""Sampling: greedy + temperature.

Day-one rule 4 (ROADMAP.md): per-request sampler state must be clonable,
because fork (M3) clones it. Hence sampling state is a per-request
``torch.Generator`` owned by the loop and re-seedable/clonable via
``get_state``/``set_state`` — never global RNG, whose consumption order
would depend on batch composition.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplingParams:
    """Frozen per-request sampling knobs. ``temperature <= 0`` means greedy.

    Top-p is deliberately deferred: M1's parity gate is greedy/temperature
    only, and every knob added here must survive the clonability rule.
    """

    temperature: float = 0.0
    seed: int | None = None
    stop_token_ids: frozenset[int] = frozenset()


def make_generator(params: SamplingParams, device: torch.device) -> torch.Generator | None:
    """Per-request RNG; None for greedy (no state to clone)."""
    if params.temperature <= 0.0:
        return None
    gen = torch.Generator(device=device)
    if params.seed is not None:
        gen.manual_seed(params.seed)
    return gen


def sample_token(
    logits: torch.Tensor,
    params: SamplingParams,
    generator: torch.Generator | None = None,
) -> int:
    """Sample one token id from a 1-D logits row (fp32)."""
    if params.temperature <= 0.0:
        return int(torch.argmax(logits).item())
    probs = torch.softmax(logits / params.temperature, dim=-1)
    return int(torch.multinomial(probs, num_samples=1, generator=generator).item())
