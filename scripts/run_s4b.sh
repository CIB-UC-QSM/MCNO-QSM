#!/usr/bin/env bash
# Run the S4b experiments one after another (server). Skips finished runs, resumes interrupted ones.
# Usage: nohup bash scripts/run_s4b.sh > runs/s4b_all.log 2>&1 &
#        optional names as arguments, e.g. bash scripts/run_s4b.sh combo mix
cd "$(dirname "$0")/.." || exit 1
names=("$@")
if [ ${#names[@]} -eq 0 ]; then
  names=(combo mix noise spheres_chi jumps tissue sti_offdiag spheres_phase cylinders)
fi
for name in "${names[@]}"; do
  out="runs/s4b_${name}"
  if [ -f "$out/DONE" ]; then echo "skip $name (DONE)"; continue; fi
  extra=""
  if [ -f "$out/last.pt" ]; then extra="--resume"; fi
  echo "=== $name $(date) $extra"
  .venv/bin/python -m mcno_qsm.train rc1 --config "configs/s4b_${name}.toml" $extra > "runs/s4b_${name}.log" 2>&1
  echo "=== $name finished $(date): $(ls $out | tr '\n' ' ')"
done
