#!/usr/bin/env bash
# Normalized layers (CLAUDE.md 9bd), server: train noise_clean_norm (mask-aware GroupNorm, pre-norm, in the Dipolelet blocks), score it alone as the extra
# column Bn of cascade_gate.py (B = noise_clean, AB = passive cascade for reference), then the scale-robustness check, pack the results.
# Usage: nohup bash scripts/after_norm.sh > runs/after_norm.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
export MCNO_VRAM_GB=10
export PYTORCH_ALLOC_CONF=expandable_segments:True
bash scripts/run_s4b.sh noise_clean_norm
.venv/bin/python scripts/test_scale_robustness.py --runs noise_clean noise_clean_norm > runs/scale_robustness.log 2>&1
.venv/bin/python scripts/cascade_gate.py --passive --bg --norm-run noise_clean_norm --out runs/cascade_norm > runs/cascade_norm.log 2>&1
tar czf ~/mcno/results_norm.tgz runs/s4b_noise_clean_norm/eval.csv runs/s4b_noise_clean_norm/history.csv runs/s4b_noise_clean_norm/metadata.json runs/s4b_noise_clean_norm/DONE runs/scale_robustness/table.txt runs/cascade_norm/fixed_meta.json runs/cascade_norm/table.txt runs/scale_robustness.log runs/cascade_norm.log runs/s4b_noise_clean_norm.log
echo "after_norm finished $(date)"
