#!/usr/bin/env bash
# S4c run 1 (CLAUDE.md 9ab), server: train ahead_noise_clean (RC1 + AHEAD), clean the generalization fields with noise_clean and it,
# rescore at fixed eNDI iterations, pack the small result files.
# Usage: nohup bash scripts/after_ahead.sh > runs/after_ahead.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
export MCNO_VRAM_GB=10
bash scripts/run_s4b.sh ahead_noise_clean
.venv/bin/python scripts/export_generalization.py --runs noise_clean ahead_noise_clean --endi none --out runs/generalization_ah > runs/generalization_ah.log 2>&1
.venv/bin/python scripts/rescore_fixed.py --dir runs/generalization_ah --methods dirty noise_clean ahead_noise_clean --real-runs noise_clean ahead_noise_clean > runs/rescore_fixed_ah.log 2>&1
tar czf ~/mcno/results_ah.tgz runs/s4b_ahead_noise_clean/eval.csv runs/s4b_ahead_noise_clean/metadata.json runs/s4b_ahead_noise_clean/DONE runs/generalization_ah/fixed_meta.json runs/generalization_ah.log runs/rescore_fixed_ah.log runs/s4b_ahead_noise_clean.log
echo "after_ahead finished $(date)"
