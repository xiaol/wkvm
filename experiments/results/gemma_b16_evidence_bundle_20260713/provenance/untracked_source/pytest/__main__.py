"""Fallback `python -m pytest` runner.

If real pytest is installed outside this repository, delegate to it. Otherwise
run unittest discovery for the file arguments used by wkvm's acceptance gates.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


def _real_pytest_main():
    here = Path(__file__).resolve().parents[1]
    for entry in sys.path[1:]:
        if not entry:
            continue
        try:
            if Path(entry).resolve() == here:
                continue
        except OSError:
            pass
        spec = importlib.util.find_spec("pytest", [entry])
        if spec and spec.origin and here not in Path(spec.origin).resolve().parents:
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            return module.main
    return None


def _module_name(path: str) -> str:
    p = Path(path)
    if p.suffix == ".py":
        p = p.with_suffix("")
    return ".".join(p.parts)


def main() -> int:
    real = _real_pytest_main()
    if real is not None:
        return int(real(sys.argv[1:]))

    verbosity = 1
    names: list[str] = []
    for arg in sys.argv[1:]:
        if arg in {"-q", "--quiet"}:
            verbosity = 1
        elif arg.startswith("-"):
            continue
        else:
            names.append(_module_name(arg))
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()
    if names:
        for name in names:
            suite.addTests(loader.loadTestsFromName(name))
    else:
        suite.addTests(loader.discover("tests"))
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
