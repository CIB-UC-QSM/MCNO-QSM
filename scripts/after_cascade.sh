#!/usr/bin/env bash
# Gated cascade (CLAUDE.md 9as), server, no training: linear regime-gated Dipole-let filter (A) then noise_clean (B), scored by eNDI at fixed iterations.
# Usage: nohup bash scripts/after_cascade.sh > runs/after_cascade.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
export MCNO_VRAM_GB=10
.venv/bin/python scripts/cascade_gate.py $CASCADE_ARGS > runs/${CASCADE_TAG:-cascade}.log 2>&1
tar czf ~/mcno/results_${CASCADE_TAG:-cascade}.tgz runs/${CASCADE_TAG:-cascade}/fixed_meta.json runs/${CASCADE_TAG:-cascade}/table.txt runs/${CASCADE_TAG:-cascade}.log
echo "after_cascade finished $(date)"
