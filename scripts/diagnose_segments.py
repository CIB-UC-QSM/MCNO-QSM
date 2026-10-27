"""Per-Dipole-let-segment diagnosis on real Sim2: how big is each segment of the true field, how much of each segment's ERROR is left
after cleaning, and how much does that error hurt eNDI? (CLAUDE.md 9ah). No training.

Notation: b = mask * D*chi (the clean target), e_m = mask * (field_m - b) the error of method m, e_m,s its image-domain component in
segment s of the Sim2 Dipole-let transform (SIM2_DIPOLELET: 3 scales x 3 |D| bands + coarse; band 0 = core near the cone, band 2 = far
from it). The 10 components sum exactly to e_m.
Per segment and method it reports
  * share of the true field, ||b_s||^2 / ||b||^2 (segments overlap, so the shares do not sum to exactly 1);
  * relative error ||e_m,s|| / ||b|| and the fraction left after cleaning, ||e_clean,s|| / ||e_dirty,s||;
  * eNDI damage of the segment error alone: NRMSE(eNDI(b + e_m,s)) - NRMSE(eNDI(b)), 25 iterations (the usual fixed-iteration score);
  * eNDI sensitivity g_s: the same difference for a perturbation of FIXED size rho*||b|| along the direction of the dirty error in
    segment s (points of NRMSE per rho of relative field error in that segment).
eNDI itself is untouched (ndiCG, alpha 1e-5, weight wPrime, 256^3 padding). Run (server):
  MCNO_VRAM_GB=10 nohup .venv/bin/python scripts/diagnose_segments.py --methods noise_clean > runs/segments.log 2>&1 &
Writes runs/segments/results.json (resumable) and runs/segments/table.txt.
With `--set rc1` the same measurement is made on the TRAINING head RC1 (TE 25 ms, 2.89 T, pad 192, weight = magn, dirty = the noise
family at SNR 60 from a fixed seed; no cleaned methods): its sensitivities g_s are the ones allowed to set a training loss, because
Sim2 must never inform training (CLAUDE.md 9aj). Output default runs/segments_rc1.
"""

import argparse
import contextlib
import io
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from mcno_qsm.dipolelets import SIM2_DIPOLELET, DipoleletTransform, image_components
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
    return float(out["rmse_history"][-1])


def load_set(name, args):
    """mask, chi, weight, voxel, oracle field b, {'dirty': field}, and the eNDI settings (te, b0, pad) of a data set."""
    if name == "sim2":
        data = load_rc2(args.rc2, 7.0)
        mask, chi, weight, voxel = data["mask"].astype(np.float32), data["chi"], data["weight"], data["voxel_size"]
        fields = {"dirty": data["field"] * mask}
        for method in args.methods:
            fields[method] = nib.load(Path("runs") / f"s4b_{method}" / "best_clean.nii.gz").get_fdata(dtype=np.float32) * mask
        return mask, chi, weight, voxel, forward_field(chi, mask, voxel), fields, dict(te=8e-3, b0=7.0, pad=256)
    from mcno_qsm.artifacts import ArtifactConfig, Volume, make_dirty
    from mcno_qsm.rc1 import load_rc1

    data = load_rc1(args.rc1)
    mask, chi, weight, voxel = data["mask"].astype(np.float32), data["chi"], data["magnitude"], data["voxel_size"]
    volume = Volume.from_rc1(data, "cpu")
    cfg = ArtifactConfig(snr_low=args.snr, snr_high=args.snr)
    dirty, _ = make_dirty("noise", volume, np.random.default_rng(args.seed), torch.Generator().manual_seed(args.seed), cfg)
    return mask, chi, weight, voxel, forward_field(chi, mask, voxel), {"dirty": dirty.numpy() * mask},         dict(te=data["te"], b0=data["b0"], pad=192)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=["sim2", "rc1"], default="sim2")
    parser.add_argument("--rc1", default="dataRC1")
    parser.add_argument("--snr", type=float, default=60.0, help="SNR of the RC1 dirty field (Sim2's psi matches about 59)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rc2", default="dataRC2")
    parser.add_argument("--methods", nargs="*", default=["noise_clean"], help="runs/s4b_<name>/best_clean.nii.gz; 'dirty' is always included")
    parser.add_argument("--rho", type=float, default=0.1, help="relative size of the sensitivity perturbation")
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    if args.set == "rc1":
        args.methods = []
    args.out = args.out or Path("runs/segments" if args.set == "sim2" else "runs/segments_rc1")
    limit_vram()
    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = args.out / "results.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    mask, chi, weight, voxel, b, fields, settings = load_set(args.set, args)

    transform = DipoleletTransform(b.shape, voxel, **SIM2_DIPOLELET, dtype=torch.float32)
    names = transform.segment_names

    def parts(volume):
        return [p.numpy() for p in image_components(transform, torch.from_numpy(np.ascontiguousarray(volume, dtype=np.float32)))]

    norm = lambda v: float(np.linalg.norm(mask * v))
    nb = norm(b)
    b_parts = parts(b)
    e_parts = {m: parts((f - b) * mask) for m, f in fields.items()}

    def endi(key, field):
        if key not in cache:
            cache[key] = invert(np.ascontiguousarray(field * mask, dtype=np.float64), mask, weight, chi, args.iters, **settings)
            cache_path.write_text(json.dumps(cache))
            print(f"{key:36s} NRMSE {cache[key]:7.2f}", flush=True)
        return cache[key]

    base = endi("oracle", b)
    full = {m: endi(f"full/{m}", f) for m, f in fields.items()}
    sens = []
    for s, name in enumerate(names):
        direction = mask * e_parts["dirty"][s]
        direction = direction / max(float(np.linalg.norm(direction)), 1e-30)
        sens.append(endi(f"sens/{s}", b + args.rho * nb * direction) - base)
    damage = {m: [endi(f"seg/{m}/{s}", b + mask * e_parts[m][s]) - base for s in range(len(names))] for m in fields}

    lines = [f"eNDI NRMSE after {args.iters} iterations on {args.set}: oracle {base:.2f}; full fields: "
             + ", ".join(f"{m} {v:.2f}" for m, v in full.items()),
             f"sensitivity g_s = change of NRMSE (points) for a perturbation of {100 * args.rho:.0f}% of ||b|| along the dirty error in the segment", ""]
    head = f"{'segment':14s} {'field%':>7s} {'g_s':>6s} | " + " | ".join(f"{m:>10s}: rel.err  left  damage" for m in fields)
    lines.append(head)
    for s, name in enumerate(names):
        row = f"{name:14s} {100 * norm(b_parts[s]) ** 2 / nb ** 2:7.2f} {sens[s]:6.2f} | "
        cells = []
        for m in fields:
            err = norm(e_parts[m][s])
            left = err / max(norm(e_parts["dirty"][s]), 1e-30)
            cells.append(f"{'':>10s}  {100 * err / nb:7.2f}% {left:5.2f} {damage[m][s]:7.2f}")
        lines.append(row + " | ".join(cells))
    lines.append("")
    for m in fields:
        lines.append(f"{m}: sum of single-segment damages {sum(damage[m]):.2f} vs full-field damage {full[m] - base:.2f} (non-additive: nonlinear, overlapping segments)")
    text = "\n".join(lines)
    (args.out / "table.txt").write_text(text + "\n")
    print(text, flush=True)
    print("segment diagnosis finished", flush=True)


if __name__ == "__main__":
    main()
