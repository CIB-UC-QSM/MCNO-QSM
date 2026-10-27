"""Field-domain error of the cleaned real Sim2 around its calcification (the 9am criterion (1)), from runs/s4b_<name>/best_clean.nii.gz.

Shell = mask voxels within `--radius` mm of the voxels with Chi < -0.5 ppm. relative error = rms(out - true) / rms(true) and gain = sum(out*true) /
sum(true^2) there (dirty and noise_clean: 1.03 / 0.99 and 0.32 / 0.29 in 9ak); also the whole mask and the 3-8 mm ring. No eNDI, CPU only.
Run: python scripts/score_shell.py --runs noise_clean iv_full
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

from mcno_qsm.rc2 import load_rc2
from mcno_qsm.scoring import forward_field


def stats(out, true, region):
    o, t = out[region], true[region]
    return float(np.sqrt(((o - t) ** 2).mean() / (t ** 2).mean())), float((o * t).sum() / (t ** 2).sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["noise_clean"])
    parser.add_argument("--rc2", default="dataRC2")
    parser.add_argument("--radius", type=float, default=3.0)
    args = parser.parse_args()
    d = load_rc2(args.rc2, 7.0)
    mask, chi, voxel = d["mask"], d["chi"], d["voxel_size"]
    true = forward_field(chi, None, voxel)
    distance = ndimage.distance_transform_edt(~(chi < -0.5), sampling=voxel)
    regions = {f"<= {args.radius:g} mm": mask & (distance <= args.radius), "3-8 mm": mask & (distance > 3) & (distance <= 8), "whole mask": mask}
    fields = {"dirty": d["field"] * mask}
    for name in args.runs:
        fields[name] = nib.load(Path("runs") / f"s4b_{name}" / "best_clean.nii.gz").get_fdata(dtype=np.float32) * mask
    print("relative error / gain of the field (out vs true D*Chi)")
    print(f"{'field':16s}" + "".join(f"{r:>20s}" for r in regions))
    for name, field in fields.items():
        cells = [f"{e:.3f} / {g:.2f}" for e, g in (stats(field, true, r) for r in regions.values())]
        print(f"{name:16s}" + "".join(f"{c:>20s}" for c in cells))


if __name__ == "__main__":
    main()
