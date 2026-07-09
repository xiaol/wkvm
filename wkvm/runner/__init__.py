"""Runner package exports.

The Gemma routed-state metadata is dependency-light. RWKV runner exports remain
available when torch is installed; importing this package in a core-only
environment should still allow `wkvm.runner.gemma_state` tests to run.
"""

from wkvm.runner.gemma_state import GemmaRoutedStateBank

try:
    from wkvm.runner.gemma_runner import GemmaRoutedSpanRunner
except ModuleNotFoundError as exc:  # pragma: no cover - torch-free envs
    if exc.name not in {"torch", "transformers"}:
        raise
    GemmaRoutedSpanRunner = None  # type: ignore[assignment]

try:
    from wkvm.runner.loop import GenerationLoop
    from wkvm.runner.runner import RWKV7Runner
    from wkvm.runner.sampling import SamplingParams
    from wkvm.runner.state import RWKV7StateBank
except ModuleNotFoundError as exc:  # pragma: no cover - exercised in core-only envs
    if exc.name != "torch":
        raise
    GenerationLoop = None  # type: ignore[assignment]
    RWKV7Runner = None  # type: ignore[assignment]
    SamplingParams = None  # type: ignore[assignment]
    RWKV7StateBank = None  # type: ignore[assignment]

__all__ = [
    "GemmaRoutedStateBank",
    "GemmaRoutedSpanRunner",
    "GenerationLoop",
    "RWKV7Runner",
    "RWKV7StateBank",
    "SamplingParams",
]
