#!/usr/bin/env bash
# eNDI-score every finished S4b run that has no score.json yet (server, GPU). Safe to rerun; skips scored runs.
# Usage: nohup bash scripts/score_s4b.sh > runs/s4b_scoring.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
for dir in runs/s4b_*/; do
  name=$(basename "$dir")
  if [ ! -f "$dir/DONE" ] || [ ! -f "$dir/best_clean.nii.gz" ]; then echo "skip $name (not finished)"; continue; fi
  if [ -f "$dir/score.json" ]; then echo "skip $name (already scored)"; continue; fi
  echo "=== scoring $name $(date)"
  .venv/bin/python scripts/score_run.py --run "runs/$name" --data dataRC2 > "runs/${name}_score.log" 2>&1
  head -2 "runs/${name}_score.log"
done
echo "scoring finished $(date)"
