#!/usr/bin/env bash
# Gate suite 2 (CLAUDE.md 9aq), server: zero-training linear Dipole-let filters scored by eNDI.
# Usage: nohup bash scripts/after_gates2.sh > runs/after_gates2.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
export MCNO_VRAM_GB=10
.venv/bin/python scripts/gate_suite2.py > runs/gates2.log 2>&1
tar czf ~/mcno/results_gates2.tgz runs/gates2/results.json runs/gates2/table.txt runs/gates2.log
echo "after_gates2 finished $(date)"
