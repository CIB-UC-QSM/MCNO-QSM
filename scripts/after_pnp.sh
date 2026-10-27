#!/usr/bin/env bash
# Plug-and-play residual refinement in eNDI (CLAUDE.md 9aw), server, no training.
# Usage: nohup bash scripts/after_pnp.sh > runs/after_pnp.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
export MCNO_VRAM_GB=10
.venv/bin/python scripts/pnp_endi.py > runs/pnp.log 2>&1
tar czf ~/mcno/results_pnp.tgz runs/pnp/fixed_meta.json runs/pnp/table.txt runs/pnp.log
echo "after_pnp finished $(date)"
