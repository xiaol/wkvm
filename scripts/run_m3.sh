#!/usr/bin/env bash
# M3 verification pipeline: tests + three demos, results to experiments/results/.
# Usage: bash scripts/run_m3.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/home/xiaol/X/HRM-Text/.venv/bin/python
export HF_HUB_OFFLINE=1
OUT=experiments/results/m3_run.log
mkdir -p experiments/results

# The NTFS volume with weights/store drops on sleep; remount if needed.
if [ ! -d /run/media/xiaol/B214449214445C0B/wkvm_bench ]; then
  udisksctl mount -b /dev/nvme1n1p2 || true
fi

{
  echo "=== M3 run $(date -Is) ==="
  echo "--- unit tests (full suite) ---"
  $PY -m unittest discover -s tests -v 2>&1 | tail -14
  echo "--- demo: fleet ---"
  $PY experiments/m3_demos.py fleet --sessions 2000
  echo "--- demo: agent (real process restart) ---"
  $PY experiments/m3_demos.py agent
  echo "--- demo: parity ---"
  $PY experiments/m3_demos.py parity
  echo "=== M3 run complete $(date -Is) ==="
} 2>&1 | tee "$OUT"
