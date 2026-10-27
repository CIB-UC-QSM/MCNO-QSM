#!/usr/bin/env bash
# Regime experiment (CLAUDE.md 9ar), server: memory check, train realmix_plain (control: same data, no regime layer) and regime (12 segments + regime filter),
# clean the generalization fields with noise_clean / realmix_plain / regime, rescore at fixed eNDI iterations, inspect what the regime filter learned, pack.
# Usage: nohup bash scripts/after_regime.sh > runs/after_regime.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
export MCNO_VRAM_GB=10
.venv/bin/python scripts/measure_regime_memory.py 96 2 > runs/regime_memory.log 2>&1
bash scripts/run_s4b.sh realmix_plain regime
.venv/bin/python scripts/export_generalization.py --runs noise_clean realmix_plain regime --endi none --out runs/generalization_regime > runs/generalization_regime.log 2>&1
.venv/bin/python scripts/rescore_fixed.py --dir runs/generalization_regime --methods dirty noise_clean realmix_plain regime --real-runs noise_clean realmix_plain regime > runs/rescore_fixed_regime.log 2>&1
.venv/bin/python scripts/inspect_regime.py regime > runs/inspect_regime.log 2>&1
tar czf ~/mcno/results_regime.tgz runs/s4b_realmix_plain/eval.csv runs/s4b_realmix_plain/history.csv runs/s4b_realmix_plain/metadata.json runs/s4b_realmix_plain/DONE runs/s4b_regime/eval.csv runs/s4b_regime/history.csv runs/s4b_regime/metadata.json runs/s4b_regime/DONE runs/generalization_regime/fixed_meta.json runs/generalization_regime.log runs/rescore_fixed_regime.log runs/inspect_regime.log runs/regime_memory.log runs/s4b_realmix_plain.log runs/s4b_regime.log
echo "after_regime finished $(date)"
