#!/usr/bin/env bash
# Calcification-shell family run (CLAUDE.md 9am), server: train noise_clean_calc, clean the generalization fields, rescore at fixed
# eNDI iterations, export the fixed-iteration chi maps of the real Sim2 and pack the results (including the cleaned Sim2 volume).
# Usage: nohup bash scripts/after_calc.sh > runs/after_calc.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
export MCNO_VRAM_GB=10
bash scripts/run_s4b.sh noise_clean_calc
.venv/bin/python scripts/export_generalization.py --runs noise_clean noise_clean_calc --endi none --out runs/generalization_calc > runs/generalization_calc.log 2>&1
.venv/bin/python scripts/rescore_fixed.py --dir runs/generalization_calc --methods dirty noise_clean noise_clean_calc --real-runs noise_clean noise_clean_calc > runs/rescore_fixed_calc.log 2>&1
.venv/bin/python scripts/export_fixed_chi.py --runs noise_clean noise_clean_calc --out runs/chi_fixed25.npz > runs/chi_fixed25.log 2>&1
tar czf ~/mcno/results_calc.tgz runs/s4b_noise_clean_calc/eval.csv runs/s4b_noise_clean_calc/history.csv runs/s4b_noise_clean_calc/metadata.json runs/s4b_noise_clean_calc/DONE runs/s4b_noise_clean_calc/best_clean.nii.gz runs/s4b_noise_clean/best_clean.nii.gz runs/generalization_calc/fixed_meta.json runs/generalization_calc.log runs/rescore_fixed_calc.log runs/chi_fixed25.npz runs/chi_fixed25.log runs/s4b_noise_clean_calc.log
echo "after_calc finished $(date)"
