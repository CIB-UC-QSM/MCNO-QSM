"""One table for all S4b runs: RC1 selection, Sim2 monitoring, eNDI scores, per-family check-slab ratios, seed spread.

Run (server or PC, from the project root):  python scripts/summarize_s4b.py [--runs runs]
Needs runs/s4b_*/eval.csv (+ DONE, score.json when available). Uses only the standard library.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

FAMILIES = ["noise", "jumps", "cylinders", "spheres_phase", "tissue", "sti", "offdiag", "spheres_chi", "combo"]


def ratio(row, family):
    d = float(row[f"{family}_in_test"])
    return float(row[f"{family}_test"]) / d if d > 1e-9 else float("nan")


def load(run: Path):
    rows = list(csv.DictReader(open(run / "eval.csv")))
    done = json.loads((run / "DONE").read_text()) if (run / "DONE").exists() else None
    score = json.loads((run / "score.json").read_text()) if (run / "score.json").exists() else None
    best_step = done["best_step"] if done else int(min(rows, key=lambda r: float(r["sel_trained"]))["step"])
    best = next(r for r in rows if int(r["step"]) == best_step)
    return rows, done, score, best_step, best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    args = parser.parse_args()
    runs = sorted(p for p in args.runs.glob("s4b_*") if (p / "eval.csv").exists())
    if not runs:
        raise SystemExit("no runs with eval.csv found")
    print(f"{'run':18s} {'done':5s} {'steps':>5s} {'best':>5s} {'sel_tr':>7s} {'sel_all':>7s} | {'sim2@best':>9s} {'sim2 min(step)':>15s} {'sim2 last':>9s} | "
          f"{'phase%':>7s} {'eNDI%':>7s} {'XSIM':>6s} {'stop':>4s} | {'eNDI% (dirty outside)':>22s} {'XSIM':>6s}")
    table = {}
    for run in runs:
        rows, done, score, best_step, best = load(run)
        sims = [(float(r["sim2_nrmse"]), int(r["step"])) for r in rows if "sim2_nrmse" in r]
        low = min(sims) if sims else (float("nan"), 0)
        cm = (score or {}).get("cleaned_masked", {})
        co = (score or {}).get("cleaned_in_mask_dirty_outside", {})
        fmt = lambda d, k, f: (f % d[k]) if k in d else "-"
        print(f"{run.name[4:]:18s} {'yes' if done else 'NO':5s} {rows[-1]['step']:>5s} {best_step:>5d} {float(best['sel_trained']):7.3f} {float(best['sel_all']):7.3f} | "
              f"{float(best['sim2_nrmse']) if 'sim2_nrmse' in best else float('nan'):9.4f} {low[0]:9.4f} ({low[1]:>4d}) {float(rows[-1]['sim2_nrmse']):9.4f} | "
              f"{fmt(cm, 'phase_nrmse_vs_clean_pct', '%7.2f')} {fmt(cm, 'endi_nrmse_pct', '%7.2f')} {fmt(cm, 'endi_xsim', '%6.4f')} {fmt(cm, 'stop_iteration', '%4d')} | "
              f"{fmt(co, 'endi_nrmse_pct', '%22.2f')} {fmt(co, 'endi_xsim', '%6.4f')}")
        table[run.name[4:]] = (best, cm, co)
    base = args.runs / "s0" / "score.json"
    if base.exists():
        print("\nbaselines (same eNDI settings):")
        for name, row in json.loads(base.read_text()).items():
            if isinstance(row, dict) and "endi_nrmse_pct" in row:
                print(f"  {name:30s} phase {row['phase_nrmse_vs_clean_pct']:6.2f}%  eNDI {row['endi_nrmse_pct']:6.2f}%  XSIM {row['endi_xsim']:.4f}  stop {row['stop_iteration']}")
    print("\ncheck-slab ratio cleaned/dirty at the selected step (<1 better; the family trained on is what the run optimised):")
    print(f"{'run':18s} " + " ".join(f"{f[:8]:>8s}" for f in FAMILIES))
    for name, (best, _, _) in table.items():
        print(f"{name:18s} " + " ".join(f"{ratio(best, f):8.3f}" for f in FAMILIES))
    for group in ("mix", "noise"):
        seeds = [n for n in table if n == group or n.startswith(f"{group}_seed")]
        scored = [n for n in seeds if "endi_nrmse_pct" in table[n][1]]
        if len(scored) < 2:
            continue
        print(f"\nseed spread over {scored} (population sd shown; n is tiny, read as a rough indication):")
        for label, idx, key in (("eNDI% masked", 1, "endi_nrmse_pct"), ("XSIM masked", 1, "endi_xsim"),
                                ("phase% masked", 1, "phase_nrmse_vs_clean_pct"), ("eNDI% dirty outside", 2, "endi_nrmse_pct")):
            values = [table[n][idx][key] for n in scored if key in table[n][idx]]
            if values:
                print(f"  {label:20s} mean {statistics.mean(values):8.4f}  sd {statistics.pstdev(values):7.4f}  values {[round(v, 4) for v in values]}")


if __name__ == "__main__":
    main()
