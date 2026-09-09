"""Kernel selection for the HF-module compute graph.

transformers resolves its Gated DeltaNet / RWKV-7 kernels once, at import of
the modeling module: the ``fla`` package (flash-linear-attention, Triton) if
importable, else its pure-torch fallback. wkvm therefore decides *before* it
imports any modeling module, through :func:`select_kernels`:

- ``"auto"`` (default, also ``WKVM_KERNELS=auto``): use ``fla`` when it
  imports and CUDA is available; otherwise the torch path.
- ``"fla"``: require ``fla`` (raise if it cannot be imported).
- ``"torch"``: force the torch path even if ``fla`` is installed (CPU tests,
  A/B runs) by masking the package for this process.

``fla`` >= 0.5 applies ``torch.compile`` to a few helpers at import time,
which on torch < 2.7 fails against any Triton >= 3.3 (inductor imports a
removed ``AttrsDescriptor``). The Triton kernels themselves do not need
inductor, so :func:`import_fla` imports ``fla`` with ``torch.compile``
temporarily replaced by the identity when the real import fails that way.
This is the only place wkvm touches the kernel stack; nothing else changes
between the two paths, which is what makes them comparable.
"""

from __future__ import annotations

import importlib
import os
import sys

_STATE: dict = {"mode": None, "fla": None}


def import_fla(strict: bool = False):
    """Return the ``fla`` module or None. Retries with ``torch.compile``
    stubbed when the plain import trips over torch/Triton version skew."""
    if "fla" in sys.modules and sys.modules["fla"] is not None:
        return sys.modules["fla"]
    if sys.modules.get("fla", 0) is None:  # masked by select_kernels("torch")
        if strict:
            raise ImportError("fla is masked for this process (WKVM_KERNELS=torch)")
        return None
    try:
        return importlib.import_module("fla")
    except ImportError as first:
        import torch

        real = torch.compile

        def _identity(fn=None, *_, **__):
            return (lambda f: f) if fn is None else fn

        torch.compile = _identity
        try:
            # A half-imported package must not shadow the retry.
            for name in [n for n in sys.modules if n == "fla" or n.startswith("fla.")]:
                del sys.modules[name]
            return importlib.import_module("fla")
        except ImportError as second:
            if strict:
                raise ImportError(f"fla unavailable: {first!r}; with compile stub: {second!r}") from second
            return None
        finally:
            torch.compile = real


def select_kernels(mode: str | None = None) -> str:
    """Decide the kernel path for this process. Idempotent for the same mode;
    a different mode after modeling modules were imported is an error."""
    mode = (mode or os.environ.get("WKVM_KERNELS", "auto")).lower()
    if mode not in ("auto", "fla", "torch"):
        raise ValueError(f"WKVM_KERNELS must be auto|fla|torch, got {mode!r}")
    if _STATE["mode"] is not None:
        if _STATE["mode"] != mode:
            raise RuntimeError(f"kernels already selected as {_STATE['mode']!r}, cannot switch to {mode!r}")
        return _STATE["resolved"]
    resolved = "torch"
    if mode == "torch":
        sys.modules["fla"] = None  # type: ignore[assignment]  # importlib.import_module("fla") now raises
    else:
        import torch

        want = mode == "fla" or torch.cuda.is_available()
        fla = import_fla(strict=(mode == "fla")) if want else None
        if fla is not None:
            resolved = "fla"
        elif mode == "fla":
            raise ImportError("WKVM_KERNELS=fla but fla could not be imported")
        else:
            sys.modules["fla"] = None  # type: ignore[assignment]
    _STATE.update(mode=mode, resolved=resolved)
    return resolved


def resolved_kernels() -> str | None:
    """``"fla"`` / ``"torch"`` once selected, else None."""
    return _STATE.get("resolved")
