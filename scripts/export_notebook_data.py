"""Export everything the visualization notebook needs into one compressed .npz (run on the GPU server).

For the dirty field, the tuned low-pass, the oracle and each requested run it stores the Sim2 field fed to
eNDI and the eNDI susceptibility map, plus the eNDI convergence histories. Same eNDI settings as score_run.py.
Run: .venv/bin/python scripts/export_notebook_data.py --runs noise mix tissue combo --out runs/notebook_data.npz
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["noise", "mix", "tissue", "combo"])
    parser.add_argument("--data", default="dataRC2")
    parser.add_argument("--out", type=Path, default=Path("runs/notebook_data.npz"))
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--te", type=float, default=8e-3)
    parser.add_argument("--b0", type=float, default=7.0)
    args = parser.parse_args()

    data = load_rc2(args.data, args.b0)
    mask, chi, weight = data["mask"], data["chi"], data["weight"]
    clean = forward_field(chi, mask, data["voxel_size"])
    dirty = data["field"] * mask
    fields = {
        "dirty": dirty,
        "lowpass": masked_lowpass(dirty, mask, args.sigma),
        "oracle": clean,
    }
    for name in args.runs:
        fields[name] = nib.load(Path("runs") / f"s4b_{name}" / "best_clean.nii.gz").get_fdata(dtype=np.float32)

    arrays = {"chi_gt": chi, "mask": mask, "clean": clean, "magnitude": data["magnitude"][..., 0],
              "weight": weight, "voxel_size": np.array(data["voxel_size"])}
    meta = {}
    for name, field in fields.items():
        result = score_endi(field, chi, mask, weight, te=args.te, b0=args.b0)
        arrays[f"field_{name}"] = field
        arrays[f"chi_{name}"] = result["chi"]
        meta[name] = {"phase_nrmse": masked_nrmse(field, clean, mask), "endi_nrmse": result["nrmse"],
                      "xsim": result["xsim"], "stop": result["iterations"],
                      "nrmse_history": result["nrmse_history"], "xsim_history": result["xsim_history"]}
        print(f"{name:12s} phase {meta[name]['phase_nrmse']:6.2f}%  eNDI {result['nrmse']:6.2f}%  "
              f"XSIM {result['xsim']:.4f}  stop {result['iterations']}", flush=True)

    arrays = {k: (v.astype(np.float16) if v.dtype == np.float32 and v.ndim == 3 else v) for k, v in arrays.items()}
    arrays["meta_json"] = np.array(json.dumps(meta))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **arrays)
    print("saved", args.out, f"{args.out.stat().st_size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
