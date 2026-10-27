"""Score a cleaner run with eNDI (same settings as s0_score.py) next to the S0 baselines.

Uses the cleaned full volume saved at the best evaluation (best_clean.nii.gz, masked ppm).
Run: python scripts/score_run.py --run runs/sim2_clean --data dataRC2
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from mcno_qsm.rc2 import load_rc2
from mcno_qsm.scoring import forward_field, masked_nrmse, score_endi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", default="dataRC2")
    parser.add_argument("--baselines", type=Path, default=Path("runs/s0/score.json"))
    parser.add_argument("--te", type=float, default=8e-3)
    parser.add_argument("--b0", type=float, default=7.0)
    args = parser.parse_args()

    data = load_rc2(args.data, args.b0)
    mask, chi, weight = data["mask"], data["chi"], data["weight"]
    clean = forward_field(chi, mask, data["voxel_size"])
    cleaned = nib.load(args.run / "best_clean.nii.gz").get_fdata(dtype=np.float32)
    inputs = {
        "cleaned_masked": cleaned,
        "cleaned_in_mask_dirty_outside": cleaned * mask + data["field"] * ~mask,
    }
    report = {}
    for name, field in inputs.items():
        result = score_endi(field, chi, mask, weight, te=args.te, b0=args.b0)
        report[name] = {"phase_nrmse_vs_clean_pct": masked_nrmse(field, clean, mask),
                        "endi_nrmse_pct": result["nrmse"], "endi_xsim": result["xsim"],
                        "stop_iteration": result["iterations"]}
        print(f"{name:32s} phase {report[name]['phase_nrmse_vs_clean_pct']:7.2f}%  eNDI {result['nrmse']:7.2f}%  "
              f"XSIM {result['xsim']:.4f}  stop {result['iterations']}", flush=True)
    if args.baselines.exists():
        for name, row in json.loads(args.baselines.read_text()).items():
            if isinstance(row, dict) and "endi_nrmse_pct" in row:
                print(f"{'[S0] ' + name:32s} phase {row['phase_nrmse_vs_clean_pct']:7.2f}%  "
                      f"eNDI {row['endi_nrmse_pct']:7.2f}%  XSIM {row['endi_xsim']:.4f}  "
                      f"stop {row['stop_iteration']}")
    (args.run / "score.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
