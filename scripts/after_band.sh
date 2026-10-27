#!/usr/bin/env bash
# Overnight chain (server): wait for the training runs to finish, then score them under a 10 GB VRAM cap, summarize, diagnose, pack.
# Usage: nohup bash scripts/after_band.sh > runs/after_band.log 2>&1 &
# Safe to rerun: scoring skips runs that already have score.json. Does not touch any running script or existing result.
cd "$(dirname "$0")/.." || exit 1
export MCNO_VRAM_GB=10
echo "waiting for training to finish $(date)"
while pgrep -f "mcno_qsm.train" > /dev/null; do sleep 60; done
echo "training finished $(date); scoring"
bash scripts/score_s4b.sh > runs/s4b_scoring3.log 2>&1
tail -n 3 runs/s4b_scoring3.log
echo "summary"
.venv/bin/python scripts/summarize_s4b.py > runs/s4b_summary3.txt 2>&1
echo "diagnosis (CPU)"
.venv/bin/python scripts/diagnose_removed.py --runs noise noise_band mix mix_band mix_12k --out runs/diagnosis_band2 > runs/diagnosis_band2.log 2>&1
tar czf ~/mcno/s4b_results3.tgz $(find runs -maxdepth 2 -path 'runs/s4b_*' \( -name '*.csv' -o -name '*.json' -o -name DONE \)) $(ls runs/*.log) runs/s4b_summary3.txt runs/diagnosis_band2.json
ls -la ~/mcno/s4b_results3.tgz
echo "all done $(date)"
