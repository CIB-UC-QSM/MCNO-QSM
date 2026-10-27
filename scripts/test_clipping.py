"""Does clipping the local field (image domain or inside the Dipole-let segments) reduce calcification streaking? No training.

Real Sim2 only. Fields: dirty (Frequency.nii masked), the cleaned Sim2 of the given runs (runs/s4b_<name>/best_clean.nii.gz) and the
oracle (mask * D*chi). Operations applied to each field AFTER cleaning, then the SAME eNDI as always (ndiCG, alpha 1e-5, weight wPrime,
256^3, FIXED 25 iterations, no change of weight or settings):
  none        the field as is
  hard<T>     clip(field, -T, T) in ppm, then mask
  dl<k>       Dipole-let segments (Sim2 parameters) -> image-domain component per detail segment, clipped at +-k * sigma_i
              (sigma_i = 1.4826 MAD of the component inside the mask), the coarse component untouched, summed, masked
The oracle rows are the control: if clipping hurts the oracle it destroys true signal.
Metrics (chi estimate vs Chi, percent): whole mask, calcification ROI (Chi < -0.5 ppm dilated by --roi-dilate mm, inside the mask),
and the rest of the mask (streak propagation away from the calcification). Also the fraction of mask voxels / field energy changed.
Run (server): MCNO_VRAM_GB=10 nohup .venv/bin/python scripts/test_clipping.py --runs noise noise_clean > runs/clipping.log 2>&1 &
Writes runs/clipping/results.json (resumable) and runs/clipping/table.txt.
"""

import argparse
import contextlib
import io
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

from mcno_qsm.dipolelets import SIM2_DIPOLELET, DipoleletTransform
from mcno_qsm.rc2 import load_rc2, phase_scale
from mcno_qsm.scoring import _modules, _pad_corner, forward_field, limit_vram


def clip_hard(field, mask, limit):
    return np.clip(field, -limit, limit) * mask


def dipolelet_components(field, voxel_size):
    transform = DipoleletTransform(field.shape, voxel_size, **SIM2_DIPOLELET, dtype=torch.float32)
    x = torch.from_numpy(field)[None, None]
    return [torch.fft.ifftn(s, dim=(-3, -2, -1), norm="ortho").real[0, 0].numpy() for s in transform.analysis(x)]


def clip_dipolelet(field, mask, voxel_size, k):
    components = dipolelet_components(field.astype(np.float32), voxel_size)
    inside = mask > 0
    total = components[-1].copy()                      # coarse component stays
    for c in components[:-1]:
        sigma = 1.4826 * np.median(np.abs(c[inside] - np.median(c[inside])))
        total += np.clip(c, -k * sigma, k * sigma)
    return total * mask


def invert(field, mask, weight, chi, iters, te=8e-3, b0=7.0, pad=256, alpha=1e-5):
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
    crop = tuple(slice(0, n) for n in field.shape)
    return out["x"][crop] / scale, float(out["xsim_history"][-1])


def nrmse(estimate, reference, region):
    return float(100 * np.linalg.norm(region * (estimate - reference)) / np.linalg.norm(region * reference))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rc2", default="dataRC2")
    parser.add_argument("--runs", nargs="+", default=["noise", "noise_clean"])
    parser.add_argument("--hard", type=float, nargs="+", default=[0.05, 0.1, 0.15])
    parser.add_argument("--dl", type=float, nargs="+", default=[5.0, 10.0, 20.0])
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--roi-threshold", type=float, default=-0.5)
    parser.add_argument("--roi-dilate", type=float, default=8.0)
    parser.add_argument("--save", action="store_true", help="also store each chi estimate as float16 .npy")
    parser.add_argument("--out", type=Path, default=Path("runs/clipping"))
    args = parser.parse_args()
    limit_vram()
    args.out.mkdir(parents=True, exist_ok=True)
    meta_path = args.out / "results.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    data = load_rc2(args.rc2, 7.0)
    mask, chi, weight, voxel = data["mask"].astype(np.float32), data["chi"], data["weight"], data["voxel_size"]
    calc = (chi < args.roi_threshold) & (mask > 0)
    distance = ndimage.distance_transform_edt(~calc, sampling=voxel)
    roi = ((distance <= args.roi_dilate) & (mask > 0)).astype(np.float32)
    rest = mask * (1 - roi)
    print(f"calcification voxels (Chi < {args.roi_threshold}): {int(calc.sum())}; ROI voxels {int(roi.sum())} "
          f"of {int(mask.sum())} in the mask", flush=True)

    fields = {"dirty": data["field"] * mask, "oracle": forward_field(chi, mask, voxel)}
    for name in args.runs:
        fields[name] = nib.load(Path("runs") / f"s4b_{name}" / "best_clean.nii.gz").get_fdata(dtype=np.float32) * mask
    operations = [("none", lambda f: f)]
    operations += [(f"hard{t:g}", lambda f, t=t: clip_hard(f, mask, t)) for t in args.hard]
    operations += [(f"dl{k:g}", lambda f, k=k: clip_dipolelet(f, mask, voxel, k)) for k in args.dl]

    rows = []
    for method, field in fields.items():
        for op_name, op in operations:
            key = f"{method}/{op_name}"
            if key not in meta:
                clipped = op(field).astype(np.float32)
                changed = np.abs(clipped - field) > 1e-7
                chi_hat, xsim = invert(clipped.astype(np.float64), mask, weight, chi, args.iters)
                meta[key] = {
                    "global": nrmse(chi_hat, chi, mask), "calc_roi": nrmse(chi_hat, chi, roi), "rest": nrmse(chi_hat, chi, rest),
                    "xsim": xsim,
                    "frac_changed": float(changed[mask > 0].mean()),
                    "energy_removed": float(np.linalg.norm((field - clipped) * mask) ** 2 / np.linalg.norm(field * mask) ** 2),
                    "field_nrmse_vs_oracle": nrmse(clipped, fields["oracle"], mask),
                }
                meta_path.write_text(json.dumps(meta))
                if args.save:
                    np.save(args.out / f"chi_{method}_{op_name}.npy", chi_hat.astype(np.float16))
            m = meta[key]
            print(f"{key:24s} global {m['global']:6.2f}  calcROI {m['calc_roi']:6.2f}  rest {m['rest']:6.2f}  XSIM {m['xsim']:.4f}  "
                  f"changed {100 * m['frac_changed']:.3f}%  energy removed {100 * m['energy_removed']:.3f}%  "
                  f"field err vs oracle {m['field_nrmse_vs_oracle']:.2f}%", flush=True)
            rows.append(f"{key:24s} {m['global']:7.2f} {m['calc_roi']:8.2f} {m['rest']:7.2f} {m['xsim']:7.4f} "
                        f"{100 * m['frac_changed']:8.3f} {100 * m['energy_removed']:8.3f}")
    header = f"{'field/operation':24s} {'global':>7s} {'calcROI':>8s} {'rest':>7s} {'XSIM':>7s} {'chg%vox':>8s} {'chg%en':>8s}"
    (args.out / "table.txt").write_text("\n".join([header, *rows]) + "\n")
    print("clipping test finished", flush=True)


if __name__ == "__main__":
    main()
