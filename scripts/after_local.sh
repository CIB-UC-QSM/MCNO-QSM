#!/usr/bin/env bash
# Option 2 follow-up (server): train noise_local seeds 43/44, score every unscored run, summarize, diagnose, pack.
# Usage: nohup bash scripts/after_local.sh > runs/after_local.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
export MCNO_VRAM_GB=10
bash scripts/run_s4b.sh noise_local_seed43 noise_local_seed44
bash scripts/score_s4b.sh
.venv/bin/python scripts/summarize_s4b.py > runs/s4b_summary5.txt
.venv/bin/python scripts/diagnose_removed.py --runs noise noise_local noise_local_seed43 mix --out runs/diagnosis_local > runs/diagnosis_local.log 2>&1
tar czf ~/mcno/s4b_results5.tgz $(find runs -maxdepth 2 -path 'runs/s4b_noise_local*' \( -name '*.csv' -o -name '*.json' -o -name DONE \)) runs/s4b_noise_local*.log runs/s4b_summary5.txt runs/diagnosis_local.log runs/diagnosis_local.json
echo "after_local finished $(date)"
