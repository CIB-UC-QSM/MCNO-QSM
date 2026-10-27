"""Score a hold-out run (S2b) with eNDI on the test slab only, next to baselines built the same way.

Every input is the dirty unmasked field everywhere except inside the test slab, where it is replaced by the
variant under test (cleaned / low-pass / oracle, keeping the dirty field outside the brain mask). The
inversion therefore sees identical data outside the slab, and chi NRMSE is computed over mask & test slab.
The low-pass sigma is chosen on the training region, never on the test slab.
Run: python scripts/score_holdout.py --run runs/sim2_holdout --data dataRC2
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter

from mcno_qsm.rc2 import load_rc2
from mcno_qsm.scoring import forward_field, masked_nrmse, score_endi


def masked_lowpass(field, mask, sigma):
    weights = gaussian_filter(mask.astype(np.float32), sigma)
    return gaussian_filter(field * mask, sigma) / np.maximum(weights, 1e-3) * mask


def region(shape, axis, bounds):
    out = np.zeros(shape, bool)
    index = [slice(None)] * 3
    index[axis] = slice(*bounds)
    out[tuple(index)] = True
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data", default="dataRC2")
    parser.add_argument("--te", type=float, default=8e-3)
    parser.add_argument("--b0", type=float, default=7.0)
    args = parser.parse_args()

    meta = json.loads((args.run / "metadata.json").read_text())
    hold = meta["holdout"]
    if hold is None:
        raise SystemExit("this run has no hold-out; use scripts/score_run.py")
    data = load_rc2(args.data, args.b0)
    mask, chi, weight, dirty = data["mask"], data["chi"], data["weight"], data["field"]
    inside = mask.astype(bool)
    train = region(mask.shape, hold["axis"], hold["train"]) & inside
    test = region(mask.shape, hold["axis"], hold["test"])
    test_mask = (test & inside).astype(np.float32)
    clean = forward_field(chi, mask, data["voxel_size"])
    cleaned = nib.load(args.run / "best_clean.nii.gz").get_fdata(dtype=np.float32)

    sigmas = [0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    sigma = min(sigmas, key=lambda s: masked_nrmse(masked_lowpass(dirty * mask, mask, s), clean, train))
    lowpass = masked_lowpass(dirty * mask, mask, sigma)

    def splice(variant):
        """Dirty everywhere; inside the test slab: `variant` in the brain, dirty outside it."""
        field = dirty.copy()
        take = test & inside
        field[take] = variant[take]
        return field

    inputs = {
        "dirty_unmasked": dirty,
        f"lowpass_sigma{sigma}": splice(lowpass),
        "cleaned": splice(cleaned),
        "oracle_in_mask": splice(clean),
    }
    report = {"holdout": hold, "lowpass_sigma_chosen_on_train": sigma}
    for name, field in inputs.items():
        result = score_endi(field, chi, mask, weight, te=args.te, b0=args.b0)
        row = {"phase_nrmse_test_pct": masked_nrmse(field, clean, test_mask),
               "endi_nrmse_test_pct": masked_nrmse(result["chi"], chi, test_mask),
               "endi_nrmse_full_mask_pct": result["nrmse"], "stop_iteration": result["iterations"]}
        report[name] = row
        print(f"{name:28s} phase(test) {row['phase_nrmse_test_pct']:7.2f}%  "
              f"eNDI(test) {row['endi_nrmse_test_pct']:7.2f}%  eNDI(full) {row['endi_nrmse_full_mask_pct']:7.2f}%  "
              f"stop {row['stop_iteration']}", flush=True)
    (args.run / "score_holdout.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
