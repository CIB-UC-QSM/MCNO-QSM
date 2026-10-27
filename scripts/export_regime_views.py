"""Fields and 25-iteration eNDI maps for the before/after notebook of the regime run (CLAUDE.md 9ar).

For each experiment: the dirty field and the field cleaned by each given run (read from runs/generalization_regime/<tag>_<exp>.npz, keys
field_<method>), the susceptibility map of each (ndiCG, alpha 1e-5, 25 fixed iterations, same inversion as rescore_fixed.py) and the oracle.
The real Sim2 (`sim2_real`) uses Frequency.nii and runs/s4b_<run>/best_clean.nii.gz. Everything is cropped to the brain bounding box (+4 voxels).
Run (server): MCNO_VRAM_GB=10 nohup .venv/bin/python scripts/export_regime_views.py > runs/export_regime_views.log 2>&1 &
Writes runs/regime_views.npz (float16; keys <exp>__field_<m>, <exp>__chi_<m>, <tag>__{chi_gt,mask,clean,chi_oracle,bbox,voxel}).
"""

import argparse
import contextlib
import io
from pathlib import Path

import nibabel as nib
import numpy as np

from mcno_qsm.rc2 import load_rc2, phase_scale
from mcno_qsm.scoring import _modules, _pad_corner, limit_vram

SETTINGS = {"rc1": (25e-3, 2.8936, 192), "sim2": (8e-3, 7.0, 256)}


def invert(field, chi, mask, weight, tag, iters, alpha=1e-5):
    te, b0, pad = SETTINGS[tag]
    tools, endi = _modules()
    scale = phase_scale(te, b0)
    params = {
        "K": tools.dipole_kernel((pad,) * 3, (pad,) * 3), "input": _pad_corner(field * scale, pad), "alpha": alpha,
        "isNonlinear": False, "weight": _pad_corner(weight, pad), "mask": _pad_corner(mask, pad),
        "maxOuterIter": iters, "tolUpdate": 0.0, "isPrecond": False,
        "chi_cosmos": _pad_corner(chi * scale, pad), "isXSIM": False, "scale": 1.0,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        out = endi.ndiCG(params)
    return out["x"][tuple(slice(0, n) for n in field.shape)] / scale


def bounding_box(mask, margin=4):
    idx = [np.where(mask.any(axis=tuple(j for j in range(3) if j != i)))[0] for i in range(3)]
    return tuple(slice(max(int(a[0]) - margin, 0), min(int(a[-1]) + margin + 1, n)) for a, n in zip(idx, mask.shape))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("runs/generalization_regime"))
    parser.add_argument("--rc2", default="dataRC2")
    parser.add_argument("--runs", nargs="+", default=["noise_clean", "realmix_plain", "regime"])
    parser.add_argument("--rc1-exps", nargs="*", default=["exp1", "exp3", "exp7", "exp8", "exp9"])
    parser.add_argument("--sim2-exps", nargs="*", default=["sim2_snr40"], help="forward-simulated Sim2 sets besides the real one")
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--out", type=Path, default=Path("runs/regime_views.npz"))
    args = parser.parse_args()
    limit_vram()
    saved = {}

    def put(key, array, box):
        saved[key] = np.ascontiguousarray(array[box]).astype(np.float16)

    for tag in ("rc1", "sim2"):
        c = np.load(args.dir / f"{tag}_common.npz")
        chi, mask, clean, weight = (c[k].astype(np.float32) for k in ("chi_gt", "mask", "clean", "weight"))
        box = bounding_box(mask)
        saved[f"{tag}__bbox"] = np.array([[s.start, s.stop] for s in box])
        saved[f"{tag}__voxel"] = c["voxel_size"]
        for key, vol in (("chi_gt", chi), ("mask", mask), ("clean", clean * mask)):
            put(f"{tag}__{key}", vol, box)
        put(f"{tag}__chi_oracle", invert(clean * mask, chi, mask, weight, tag, args.iters), box)
        print(tag, "oracle done", flush=True)

        fields, weights = {}, {}                       # experiment -> {method: field}, experiment -> eNDI weight
        exps = args.rc1_exps if tag == "rc1" else args.sim2_exps
        for exp in exps:
            z = np.load(args.dir / f"{tag}_{exp}.npz")
            weights[exp] = z["weight"].astype(np.float32) if "weight" in z.files else weight
            fields[exp] = {m: z[f"field_{m}"].astype(np.float32) * mask for m in ["dirty", *args.runs] if f"field_{m}" in z.files}
        if tag == "sim2":
            data = load_rc2(args.rc2, 7.0)
            real = {"dirty": data["field"] * mask}
            for name in args.runs:
                real[name] = nib.load(Path("runs") / f"s4b_{name}" / "best_clean.nii.gz").get_fdata(dtype=np.float32) * mask
            fields["sim2_real"], weights["sim2_real"] = real, data["weight"]
        for exp, by_method in fields.items():
            wt = weights[exp]
            for method, field in by_method.items():
                put(f"{exp}__field_{method}", field, box)
                x = invert(np.ascontiguousarray(field, dtype=np.float64), chi, mask, wt, tag, args.iters)
                put(f"{exp}__chi_{method}", x * mask, box)
                err = 100 * np.linalg.norm((x - chi)[mask > 0]) / np.linalg.norm(chi[mask > 0])
                print(f"{exp:12s} {method:16s} eNDI NRMSE {err:6.2f}%", flush=True)
    np.savez_compressed(args.out, **saved)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
