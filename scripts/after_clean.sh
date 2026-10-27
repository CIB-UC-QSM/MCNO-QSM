#!/usr/bin/env bash
# Identity-family run (CLAUDE.md 9x), server: train noise_clean, clean the generalization fields with it and with `noise`,
# rescore at fixed eNDI iterations, pack the small result files.
# Usage: nohup bash scripts/after_clean.sh > runs/after_clean.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
export MCNO_VRAM_GB=10
bash scripts/run_s4b.sh noise_clean
.venv/bin/python scripts/export_generalization.py --runs noise noise_clean --endi none --out runs/generalization_nc > runs/generalization_nc.log 2>&1
.venv/bin/python scripts/rescore_fixed.py --dir runs/generalization_nc --methods dirty noise noise_clean --real-runs noise noise_clean > runs/rescore_fixed_nc.log 2>&1
tar czf ~/mcno/results_nc.tgz runs/s4b_noise_clean/eval.csv runs/s4b_noise_clean/metadata.json runs/s4b_noise_clean/DONE runs/generalization_nc/fixed_meta.json runs/generalization_nc.log runs/rescore_fixed_nc.log runs/s4b_noise_clean.log
echo "after_clean finished $(date)"
