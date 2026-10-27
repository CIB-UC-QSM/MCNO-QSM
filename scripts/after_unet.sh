#!/usr/bin/env bash
# Option 1 (server): train noise_unet seeds 42/43/44, score, ablate the branch, summarize, pack.
# Usage: nohup bash scripts/after_unet.sh > runs/after_unet.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
export MCNO_VRAM_GB=10
bash scripts/run_s4b.sh noise_unet noise_unet_seed43 noise_unet_seed44
bash scripts/score_s4b.sh
.venv/bin/python scripts/ablate_unet.py --runs s4b_noise_unet s4b_noise_unet_seed43 s4b_noise_unet_seed44 > runs/ablate_unet.log 2>&1
for r in s4b_noise_unet s4b_noise_unet_seed43 s4b_noise_unet_seed44; do
  .venv/bin/python scripts/score_run.py --run runs/${r}_nobranch --data dataRC2 > runs/${r}_nobranch_score.log 2>&1
done
.venv/bin/python scripts/summarize_s4b.py > runs/s4b_summary6.txt
.venv/bin/python scripts/diagnose_removed.py --runs noise noise_unet noise_unet_seed43 noise_unet_seed44 --out runs/diagnosis_unet > runs/diagnosis_unet.log 2>&1
tar czf ~/mcno/s4b_results6.tgz $(find runs -maxdepth 2 -path 'runs/s4b_noise_unet*' \( -name '*.csv' -o -name '*.json' -o -name DONE \)) runs/s4b_noise_unet*.log runs/s4b_summary6.txt runs/ablate_unet.* runs/diagnosis_unet.log runs/diagnosis_unet.json
echo "after_unet finished $(date)"
