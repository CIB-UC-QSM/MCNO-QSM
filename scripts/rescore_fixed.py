"""Fixed-iteration eNDI (ndiCG, NO automatic stopping) on the exported fields of export_generalization.py.

The auto-stop fires at very different iterations for different fields (9-150), which confounds comparisons. Here every
field gets the same number of iterations and the whole NRMSE / XSIM curve is stored; read the table at 10, 25, 40.
Also scores the REAL Sim2 (Frequency.nii masked, and best_clean.nii.gz of the given runs) as experiment `sim2_real`.
Run (server): MCNO_VRAM_GB=10 nohup .venv/bin/python scripts/rescore_fixed.py --real-runs noise mix > runs/rescore_fixed.log 2>&1 &
Writes runs/generalization/fixed_meta.json (resumable: finished entries are skipped).
"""

import argparse
import contextlib
import io
import json
from pathlib import Path

import nibabel as nib
import numpy as np

from mcno_qsm.rc2 import load_rc2, phase_scale
from mcno_qsm.scoring import _modules, _pad_corner, limit_vram, masked_nrmse

SHOW = (10, 25, 40, 100)


def invert(field, chi, mask, weight, te, b0, pad, iters, alpha=1e-5):
    tools, endi = _modules()
    scale = phase_scale(te, b0)
    params = {
        "K": tools.dipole_kernel((pad,) * 3, (pad,) * 3), "input": _pad_corner(field * scale, pad), "alpha": alpha,
        "isNonlinear": False, "weight": _pad_corner(weight, pad), "mask": _pad_corner(mask, pad),
        "maxOuterIter": iters, "tolUpdate": 0.0, "isPrecond": False,
        "chi_cosmos": _pad_corner(chi * scale, pad), "isXSIM": True, "scale": 1.0,
    }
    with contextlib.redirect_stdout(io.StringIO()):
        out = endi.ndiCG(params)
    return {"nrmse": [float(v) for v in out["rmse_history"]], "xsim": [float(v) for v in out["xsim_history"]]}


def line(label, h):
    g = lambda i: h["nrmse"][min(i, len(h["nrmse"]) - 1)]
    return (f"{label:28s} " + " ".join(f"@{i}: {g(i):6.2f}%" for i in SHOW if i < len(h["nrmse"]))
            + f"   min {min(h['nrmse']):6.2f}% @{h['nrmse'].index(min(h['nrmse']))}   XSIM@25 {h['xsim'][min(25, len(h['xsim']) - 1)]:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("runs/generalization"))
    parser.add_argument("--sets", nargs="+", default=["rc1", "sim2"])
    parser.add_argument("--methods", nargs="+", default=["dirty", "noise", "mix"])
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--pad-rc1", type=int, default=192)
    parser.add_argument("--real-runs", nargs="*", default=[], help="also score the real Sim2 cleaned by these runs")
    parser.add_argument("--rc2", default="dataRC2")
    parser.add_argument("--only", nargs="+", default=None)
    args = parser.parse_args()
    limit_vram()
    meta_path = args.dir / "fixed_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    def run(tag, exp, method, field, common, weight, te, b0, pad):
        key = f"{tag}/{exp}/{method}"
        if key in meta:
            return
        meta[key] = invert(field.astype(np.float32), common["chi_gt"].astype(np.float32), common["mask"].astype(np.float32),
                           weight.astype(np.float32), te, b0, pad, args.iters)
        meta_path.write_text(json.dumps(meta))
        print(line(f"{tag} {exp} {method}", meta[key]), flush=True)

    for tag in args.sets:
        te, b0, pad = (25e-3, 2.8936, args.pad_rc1) if tag == "rc1" else (8e-3, 7.0, 256)
        c = np.load(args.dir / f"{tag}_common.npz")
        common = {k: c[k] for k in c.files}
        weight_common = common["weight"].astype(np.float32)
        mask = common["mask"].astype(np.float32)
        run(tag, "oracle", "oracle", common["clean"].astype(np.float32) * mask, common, weight_common, te, b0, pad)
        for file in sorted(args.dir.glob(f"{tag}_*.npz")):
            exp = file.stem[len(tag) + 1:]
            if exp in ("common", "oracle") or exp == "exp10" or (args.only and exp not in args.only):
                continue                        # exp10 is exp1 (the net masks the input anyway)
            z = np.load(file)
            weight = z["weight"].astype(np.float32) if "weight" in z.files else weight_common
            for method in args.methods:
                if f"field_{method}" in z.files:
                    run(tag, exp, method, z[f"field_{method}"], common, weight, te, b0, pad)
        if tag == "sim2" and args.real_runs:
            data = load_rc2(args.rc2, 7.0)
            fields = {"dirty": data["field"] * data["mask"]}
            for name in args.real_runs:
                fields[name] = nib.load(Path("runs") / f"s4b_{name}" / "best_clean.nii.gz").get_fdata(dtype=np.float32)
            for method, field in fields.items():
                run(tag, "sim2_real", method, field, common, data["weight"], te, b0, pad)
    print("rescore finished", flush=True)


if __name__ == "__main__":
    main()
