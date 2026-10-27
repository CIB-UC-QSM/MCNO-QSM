#!/usr/bin/env bash
# Dipole-let gain network (CLAUDE.md 9az), server: train s4b_gainnet (3D U-Net of passive gains, trained through the frozen noise_clean), then score it with
# cascade_gate.py as extra columns GNa / GN next to dirty, B, A, AB (passive g1; no remnant stress tests), pack the results.
# Usage: nohup bash scripts/after_gainnet.sh > runs/after_gainnet.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
export MCNO_VRAM_GB=10
export PYTORCH_ALLOC_CONF=expandable_segments:True     # the first launch died of fragmentation at the 10 GB cap (full-volume Sim2 monitor)
bash scripts/run_s4b.sh gainnet
.venv/bin/python scripts/cascade_gate.py --passive --bg --gain-run gainnet --out runs/cascade_gainnet > runs/cascade_gainnet.log 2>&1
tar czf ~/mcno/results_gainnet.tgz runs/s4b_gainnet/eval.csv runs/s4b_gainnet/history.csv runs/s4b_gainnet/metadata.json runs/s4b_gainnet/DONE runs/cascade_gainnet/fixed_meta.json runs/cascade_gainnet/table.txt runs/cascade_gainnet.log runs/s4b_gainnet.log
echo "after_gainnet finished $(date)"
