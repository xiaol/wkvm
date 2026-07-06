#!/usr/bin/env python3
"""M3 verification pipeline orchestrator (stdlib-only, system python3).

Runs: full unittest suite -> fleet demo -> agent restart demo -> parity demo,
teeing everything to experiments/results/m3_run.log. Exits nonzero on the
first failing stage.
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = "/home/xiaol/X/HRM-Text/.venv/bin/python"
BENCH = Path("/run/media/xiaol/B214449214445C0B/wkvm_bench")
LOG = REPO / "experiments/results/m3_run.log"
ENV = {**os.environ, "HF_HUB_OFFLINE": "1"}

log_lines = []


def emit(line: str) -> None:
    print(line, flush=True)
    log_lines.append(line)


def run(label: str, cmd: list[str], tail: int | None = None) -> int:
    emit(f"--- {label} ---")
    p = subprocess.run(cmd, cwd=REPO, env=ENV, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = p.stdout.splitlines()
    for line in (out[-tail:] if tail else out):
        emit(line)
    emit(f"--- {label}: exit {p.returncode} ---")
    return p.returncode


def main() -> int:
    emit(f"=== M3 run {datetime.now().isoformat()} ===")
    if not BENCH.is_dir():
        subprocess.run(["udisksctl", "mount", "-b", "/dev/nvme1n1p2"])
    if not BENCH.is_dir():
        emit("FATAL: bench volume not mounted")
        return 2
    stages = [
        ("unit tests", [PY, "-m", "unittest", "discover", "-s", "tests", "-v"], 16),
        ("demo fleet", [PY, "experiments/m3_demos.py", "fleet", "--sessions", "2000"], None),
        ("demo agent", [PY, "experiments/m3_demos.py", "agent"], None),
        ("demo parity", [PY, "experiments/m3_demos.py", "parity"], None),
    ]
    rc = 0
    for label, cmd, tail in stages:
        rc = run(label, cmd, tail)
        if rc != 0:
            emit(f"STOPPING: {label} failed")
            break
    emit(f"=== M3 run complete rc={rc} {datetime.now().isoformat()} ===")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(log_lines) + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())
