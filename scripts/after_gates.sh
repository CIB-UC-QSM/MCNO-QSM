#!/usr/bin/env bash
# Gate suite (CLAUDE.md 9ap), server: eNDI tests without training (shell oracle, cone notch, oracle replacement per |D| / |k| region).
# Usage: nohup bash scripts/after_gates.sh > runs/after_gates.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
export MCNO_VRAM_GB=10
.venv/bin/python scripts/gate_suite.py > runs/gates.log 2>&1
tar czf ~/mcno/results_gates.tgz runs/gates/results.json runs/gates/table.txt runs/gates.log
echo "after_gates finished $(date)"
