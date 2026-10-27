#!/usr/bin/env bash
# Dipole-let sensitivity-weighted loss run (CLAUDE.md 9aj), server: train noise_clean_seg, clean the generalization fields,
# rescore at fixed eNDI iterations, repeat the per-segment diagnosis on Sim2 for noise_clean and noise_clean_seg, pack the results.
# Usage: nohup bash scripts/after_seg.sh > runs/after_seg.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
export MCNO_VRAM_GB=10
bash scripts/run_s4b.sh noise_clean_seg
.venv/bin/python scripts/export_generalization.py --runs noise_clean noise_clean_seg --endi none --out runs/generalization_seg > runs/generalization_seg.log 2>&1
.venv/bin/python scripts/rescore_fixed.py --dir runs/generalization_seg --methods dirty noise_clean noise_clean_seg --real-runs noise_clean noise_clean_seg > runs/rescore_fixed_seg.log 2>&1
.venv/bin/python scripts/diagnose_segments.py --methods noise_clean noise_clean_seg --out runs/segments_seg > runs/segments_seg.log 2>&1
tar czf ~/mcno/results_seg.tgz runs/s4b_noise_clean_seg/eval.csv runs/s4b_noise_clean_seg/history.csv runs/s4b_noise_clean_seg/metadata.json runs/s4b_noise_clean_seg/DONE runs/generalization_seg/fixed_meta.json runs/generalization_seg.log runs/rescore_fixed_seg.log runs/segments_seg/table.txt runs/s4b_noise_clean_seg.log
echo "after_seg finished $(date)"
