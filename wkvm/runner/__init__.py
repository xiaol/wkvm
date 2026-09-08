"""Runner package exports.

The Gemma routed-state metadata is dependency-light. RWKV runner exports remain
available when torch is installed; importing this package in a core-only
environment should still allow `wkvm.runner.gemma_state` tests to run.
"""

try:
    from wkvm.runner.gemma_state import GemmaRoutedStateBank
except ModuleNotFoundError as exc:  # pragma: no cover - partial checkouts
    if exc.name != "wkvm.runner.gemma_state":
        raise
    GemmaRoutedStateBank = None  # type: ignore[assignment]

try:
    from wkvm.runner.gemma_runner import GemmaRoutedSpanRunner
except ModuleNotFoundError as exc:  # pragma: no cover - torch-free envs
    if exc.name not in {"torch", "transformers", "wkvm.runner.gemma_runner"}:
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

try:
    from wkvm.runner.hybrid_runner import Qwen35HybridRunner
    from wkvm.runner.hybrid_state import Qwen35StateBank
except ModuleNotFoundError as exc:  # pragma: no cover - torch-free envs
    if exc.name != "torch":
        raise
    Qwen35HybridRunner = None  # type: ignore[assignment]
    Qwen35StateBank = None  # type: ignore[assignment]

__all__ = [
    "GemmaRoutedStateBank",
    "GemmaRoutedSpanRunner",
    "GenerationLoop",
    "Qwen35HybridRunner",
    "Qwen35StateBank",
    "RWKV7Runner",
    "RWKV7StateBank",
    "SamplingParams",
]
