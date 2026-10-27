#!/usr/bin/env bash
# Intravoxel / acquisition-noise runs (CLAUDE.md 9bp), server: train iv_noisestat and iv_full one after another, clean the generalization fields
# and the paper tests with them and noise_clean, rescore at fixed eNDI iterations (25 and 100), weight-scale robustness, the real-Sim2 calcification
# ROI (eNDI, `none` rows of test_clipping) and the field-domain shell error, pack the small result files.
# Usage: nohup bash scripts/after_intravoxel.sh > runs/after_intravoxel.log 2>&1 &
#        optional run names as arguments (default: iv_noisestat iv_full)
# Never edit this file while it runs (bash reads scripts incrementally).
cd "$(dirname "$0")/.." || exit 1
export MCNO_VRAM_GB=10
export PYTORCH_ALLOC_CONF=expandable_segments:True
names=("$@")
if [ ${#names[@]} -eq 0 ]; then
  names=(iv_noisestat iv_full)
fi
bash scripts/run_s4b.sh "${names[@]}"
done_names=()
for n in "${names[@]}"; do
  if [ -f "runs/s4b_${n}/DONE" ]; then done_names+=("$n"); else echo "NOT DONE: $n (left out of the scoring)"; fi
done
if [ ${#done_names[@]} -eq 0 ]; then echo "nothing finished"; exit 1; fi
.venv/bin/python scripts/export_generalization.py --runs noise_clean "${done_names[@]}" --endi none --out runs/generalization_iv > runs/generalization_iv.log 2>&1
.venv/bin/python scripts/rescore_fixed.py --dir runs/generalization_iv --iters 100 --methods dirty noise_clean "${done_names[@]}" --real-runs noise_clean "${done_names[@]}" > runs/rescore_fixed_iv.log 2>&1
.venv/bin/python scripts/export_paper_tests.py --runs noise_clean "${done_names[@]}" --out runs/generalization_iv_tests > runs/iv_paper_tests.log 2>&1
.venv/bin/python scripts/rescore_fixed.py --dir runs/generalization_iv_tests --sets rc1 --iters 100 --methods dirty noise_clean "${done_names[@]}" > runs/rescore_fixed_iv_tests.log 2>&1
for tset in noise src; do
  .venv/bin/python scripts/export_iv_tests.py --set $tset --runs noise_clean "${done_names[@]}" --out runs/generalization_iv_$tset > runs/iv_tests_$tset.log 2>&1
  .venv/bin/python scripts/rescore_fixed.py --dir runs/generalization_iv_$tset --sets rc1 --iters 100 --methods dirty noise_clean "${done_names[@]}" > runs/rescore_fixed_iv_$tset.log 2>&1
done
.venv/bin/python scripts/test_scale_robustness.py --runs noise_clean "${done_names[@]}" --dir runs/generalization_iv --out runs/scale_robustness_iv > runs/scale_robustness_iv.log 2>&1
.venv/bin/python scripts/test_clipping.py --runs noise_clean "${done_names[@]}" --hard 0.15 --dl 20 --out runs/clipping_iv > runs/clipping_iv.log 2>&1
.venv/bin/python scripts/score_shell.py --runs noise_clean "${done_names[@]}" > runs/shell_iv.txt 2>&1
files=(runs/generalization_iv/fixed_meta.json runs/generalization_iv_tests/fixed_meta.json runs/generalization_iv_tests/paper_tests.json
       runs/scale_robustness_iv/table.txt runs/clipping_iv/table.txt runs/shell_iv.txt runs/generalization_iv.log runs/rescore_fixed_iv.log
       runs/iv_paper_tests.log runs/rescore_fixed_iv_tests.log runs/scale_robustness_iv.log runs/clipping_iv.log
       runs/generalization_iv_noise/iv_tests.json runs/generalization_iv_src/iv_tests.json runs/generalization_iv_noise/fixed_meta.json
       runs/generalization_iv_src/fixed_meta.json runs/iv_tests_noise.log runs/iv_tests_src.log runs/rescore_fixed_iv_noise.log runs/rescore_fixed_iv_src.log)
for n in "${done_names[@]}"; do
  files+=("runs/s4b_${n}/eval.csv" "runs/s4b_${n}/history.csv" "runs/s4b_${n}/metadata.json" "runs/s4b_${n}/DONE" "runs/s4b_${n}.log")
done
tar czf ~/mcno/results_iv.tgz "${files[@]}"
echo "after_intravoxel finished $(date)"
