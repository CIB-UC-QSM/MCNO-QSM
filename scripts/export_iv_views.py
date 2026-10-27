"""Small slice file for the 9bp notebook: real and forward-simulated Sim2, dirty / noise_clean / iv_noisestat / iv_full (server, GPU for eNDI).

Fields: the REAL Sim2 (dirty = Frequency.nii masked; cleaned = runs/s4b_<run>/best_clean.nii.gz) and the forward fields of
runs/generalization_iv (sim2_snrcal, sim2_jumps, sim2_snr40). Each field gets eNDI at a fixed iteration count (ndiCG, alpha 1e-5,
256^3 padding, default 25). Only 2D slices are stored (three planes through the brain centre and through the calcification,
float16): runs/iv_views.npz is a few MB. Run (server):
    MCNO_VRAM_GB=10 .venv/bin/python scripts/export_iv_views.py > runs/iv_views.log 2>&1
"""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np

from export_fixed_chi import invert
from mcno_qsm.rc2 import load_rc2
from mcno_qsm.scoring import forward_field, limit_vram


def planes(volume, point):
    x, y, z = (int(v) for v in point)
    return {"sag": volume[x], "cor": volume[:, y], "axi": volume[:, :, z]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rc2", default="dataRC2")
    parser.add_argument("--dir", type=Path, default=Path("runs/generalization_iv"))
    parser.add_argument("--runs", nargs="+", default=["noise_clean", "iv_noisestat", "iv_full"])
    parser.add_argument("--forward", nargs="+", default=["sim2_snrcal", "sim2_jumps", "sim2_snr40"])
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--out", type=Path, default=Path("runs/iv_views.npz"))
    args = parser.parse_args()
    limit_vram()
    data = load_rc2(args.rc2, 7.0)
    mask, chi, weight, voxel = data["mask"].astype(np.float32), data["chi"], data["weight"], data["voxel_size"]
    calc = np.argwhere(chi < -0.5)
    points = {"centre": np.argwhere(mask > 0).mean(0), "calc": calc.mean(0)}
    print("calcification voxels", len(calc), "centre", points["calc"].round(1), flush=True)

    sets = {"real": {"dirty": data["field"] * mask}}
    for run in args.runs:
        sets["real"][run] = nib.load(Path("runs") / f"s4b_{run}" / "best_clean.nii.gz").get_fdata(dtype=np.float32) * mask
    for name in args.forward:
        z = np.load(args.dir / f"sim2_{name}.npz")          # files are named <set>_<experiment>, experiment = sim2_snrcal ...
        sets[name] = {"dirty": z["field_dirty"].astype(np.float32)}
        for run in args.runs:
            sets[name][run] = z[f"field_{run}"].astype(np.float32)
    truth = forward_field(chi, mask, voxel)

    out = {"points": np.array([points["centre"], points["calc"]])}
    for pname, p in points.items():
        for plane, sl in planes(mask, p).items():
            out[f"mask/{pname}/{plane}"] = sl.astype(np.uint8)
        for plane, sl in planes(chi, p).items():
            out[f"chi_true/{pname}/{plane}"] = sl.astype(np.float16)
        for plane, sl in planes(truth, p).items():
            out[f"field_true/{pname}/{plane}"] = sl.astype(np.float16)
    for sname, fields in sets.items():
        for method, field in fields.items():
            field = np.ascontiguousarray(field, dtype=np.float64)
            x = invert(field, mask, weight, chi, args.iters)
            error = 100 * np.linalg.norm((x - chi)[mask > 0]) / np.linalg.norm(chi[mask > 0])
            print(f"{sname:12s} {method:14s} eNDI@{args.iters} NRMSE {error:6.2f}%", flush=True)
            out[f"nrmse/{sname}/{method}"] = np.float32(error)
            for pname, p in points.items():
                for plane, sl in planes(field, p).items():
                    out[f"field/{sname}/{method}/{pname}/{plane}"] = sl.astype(np.float16)
                for plane, sl in planes(x, p).items():
                    out[f"chi/{sname}/{method}/{pname}/{plane}"] = sl.astype(np.float16)
    np.savez_compressed(args.out, **out)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
