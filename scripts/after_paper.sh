#!/usr/bin/env bash
# Paper-recipe replication (CLAUDE.md 9bo), server: train the runs one after another (E3 needs E2 first), clean the generalization fields and the
# paper tests with them and noise_clean, rescore at fixed eNDI iterations (25 and 100), weight-scale robustness, pack the small result files.
# Usage: nohup bash scripts/after_paper.sh > runs/after_paper.log 2>&1 &
#        optional run names as arguments (default: the four main runs), e.g. bash scripts/after_paper.sh paper_ws paper_e4_l2only
# Never edit this file while it runs (bash reads scripts incrementally).
cd "$(dirname "$0")/.." || exit 1
export MCNO_VRAM_GB=10
export PYTORCH_ALLOC_CONF=expandable_segments:True
names=("$@")
if [ ${#names[@]} -eq 0 ]; then
  names=(paper_e1_rc1 paper_e1_ahead1 paper_e2_geo paper_e3_ft)
fi
bash scripts/run_s4b.sh "${names[@]}"
done_names=()
for n in "${names[@]}"; do
  if [ -f "runs/s4b_${n}/DONE" ]; then done_names+=("$n"); else echo "NOT DONE: $n (left out of the scoring)"; fi
done
if [ ${#done_names[@]} -eq 0 ]; then echo "nothing finished"; exit 1; fi
.venv/bin/python scripts/export_generalization.py --runs noise_clean "${done_names[@]}" --endi none --out runs/generalization_paper > runs/generalization_paper.log 2>&1
.venv/bin/python scripts/rescore_fixed.py --dir runs/generalization_paper --iters 100 --methods dirty noise_clean "${done_names[@]}" --real-runs noise_clean "${done_names[@]}" > runs/rescore_fixed_paper.log 2>&1
.venv/bin/python scripts/export_paper_tests.py --runs noise_clean "${done_names[@]}" --out runs/generalization_paper_tests > runs/paper_tests.log 2>&1
.venv/bin/python scripts/rescore_fixed.py --dir runs/generalization_paper_tests --sets rc1 --iters 100 --methods dirty noise_clean "${done_names[@]}" > runs/rescore_fixed_paper_tests.log 2>&1
.venv/bin/python scripts/test_scale_robustness.py --runs noise_clean "${done_names[@]}" --dir runs/generalization_paper --out runs/scale_robustness_paper > runs/scale_robustness_paper.log 2>&1
files=(runs/generalization_paper/fixed_meta.json runs/generalization_paper_tests/fixed_meta.json runs/generalization_paper_tests/paper_tests.json
       runs/scale_robustness_paper/table.txt runs/generalization_paper.log runs/rescore_fixed_paper.log runs/paper_tests.log
       runs/rescore_fixed_paper_tests.log runs/scale_robustness_paper.log)
for n in "${done_names[@]}"; do
  files+=("runs/s4b_${n}/eval.csv" "runs/s4b_${n}/history.csv" "runs/s4b_${n}/metadata.json" "runs/s4b_${n}/DONE" "runs/s4b_${n}.log")
done
tar czf ~/mcno/results_paper.tgz "${files[@]}"
echo "after_paper finished $(date)"
