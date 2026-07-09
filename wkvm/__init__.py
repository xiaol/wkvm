"""wkvm package exports."""

try:
    from wkvm.gemma_engine import GemmaNativeEngine
except ModuleNotFoundError as exc:  # pragma: no cover - core-only envs
    if exc.name not in {"torch", "transformers"}:
        raise
    GemmaNativeEngine = None  # type: ignore[assignment]

__all__ = ["GemmaNativeEngine"]
