"""eNDI susceptibility maps of the REAL Sim2 after a fixed number of iterations, for the notebook (no automatic stop).

Fields: dirty, the cleaned Sim2 of each given run (runs/s4b_<name>/best_clean.nii.gz) and the oracle mask * D*chi. Same eNDI as always
(ndiCG, alpha 1e-5, weight wPrime, 256^3 padding, 25 iterations). Writes runs/chi_fixed25.npz (float16 maps, keys chi_<name>) and prints
the NRMSE of each map inside the mask.
Run (server): MCNO_VRAM_GB=10 .venv/bin/python scripts/export_fixed_chi.py --runs noise_clean noise_clean_calc
"""

import argparse
import contextlib
import io
from pathlib import Path

import nibabel as nib
import numpy as np

from mcno_qsm.rc2 import load_rc2, phase_scale
from mcno_qsm.scoring import _modules, _pad_corner, forward_field, limit_vram


def invert(field, mask, weight, chi, iters, te=8e-3, b0=7.0, pad=256, alpha=1e-5):
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
    crop = tuple(slice(0, n) for n in field.shape)
    return out["x"][crop] / scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rc2", default="dataRC2")
    parser.add_argument("--runs", nargs="*", default=["noise_clean"])
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--out", type=Path, default=Path("runs/chi_fixed25.npz"))
    args = parser.parse_args()
    limit_vram()
    data = load_rc2(args.rc2, 7.0)
    mask, chi, weight, voxel = data["mask"].astype(np.float32), data["chi"], data["weight"], data["voxel_size"]
    fields = {"dirty": data["field"] * mask, "oracle": forward_field(chi, mask, voxel)}
    for name in args.runs:
        fields[name] = nib.load(Path("runs") / f"s4b_{name}" / "best_clean.nii.gz").get_fdata(dtype=np.float32) * mask
    maps = {}
    for name, field in fields.items():
        x = invert(np.ascontiguousarray(field, dtype=np.float64), mask, weight, chi, args.iters)
        maps[f"chi_{name}"] = x.astype(np.float16)
        error = 100 * np.linalg.norm((x - chi)[mask > 0]) / np.linalg.norm(chi[mask > 0])
        print(f"{name:24s} NRMSE {error:6.2f}%", flush=True)
    np.savez_compressed(args.out, **maps)
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
