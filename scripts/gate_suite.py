"""Gate suite (CLAUDE.md 9ap): eNDI-based tests with NO training, to decide what is worth building. Run on the server (GPU).

All scores are eNDI NRMSE % at 25 FIXED iterations (diagnose_segments.invert: ndiCG, alpha 1e-5, as the rest of the project).
Sets: sim2 (real Frequency.nii; dirty, noise_clean cleaned field, oracle) and rc1 exp7/exp8/exp9 (the real in-vivo-like residuals of the
paper notebook; dirty, oracle). Sim2 / RC1 truth is used for DIAGNOSIS only.
Families of fields per start (dirty, and noise_clean for sim2):
  notch d      linear cone notch: spectrum multiplied by exp(-(d/|D|)^4) (removes |D| < d, smooth), then masked.   [a non-learned baseline]
  fix <R>      oracle replacement of the TRUE field inside the Fourier region R (|D|<0.05, |D|<0.1, |D|<0.2, |D|>=0.1, |k|<0.1, |k|<0.2,
               |k|>=0.2 cycles/mm): shows how much of the gap to the oracle lives in that region = the headroom of a perfect cleaner there.
  shell r      (sim2 only) true field put back inside r mm of the 37 calcification voxels (G1).
Writes runs/gates/results.json (resumable) and runs/gates/table.txt.
Run: MCNO_VRAM_GB=10 nohup .venv/bin/python scripts/gate_suite.py > runs/gates.log 2>&1 &     Dry run (laptop): python -I scripts/gate_suite.py --dry
"""

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_segments import invert  # noqa: E402
from export_generalization import rc1_experiments  # noqa: E402

from mcno_qsm.physics import dipole_kernel  # noqa: E402
from mcno_qsm.rc1 import load_rc1  # noqa: E402
from mcno_qsm.rc2 import load_rc2  # noqa: E402
from mcno_qsm.scoring import forward_field, limit_vram  # noqa: E402

NOTCH = (0.05, 0.10, 0.15)
REGIONS = {
    "|D|<0.05": lambda D, K: D < 0.05, "|D|<0.1": lambda D, K: D < 0.1, "|D|<0.2": lambda D, K: D < 0.2, "|D|>=0.1": lambda D, K: D >= 0.1,
    "|k|<0.1": lambda D, K: K < 0.1, "|k|<0.2": lambda D, K: K < 0.2, "|k|>=0.2": lambda D, K: K >= 0.2,
}


def centre(a, n):
    s = [(m - min(m, n)) // 2 for m in a.shape[:3]]
    return a[tuple(slice(i, i + min(m, n)) for i, m in zip(s, a.shape[:3]))]


def load_sets(args):
    sets = {}
    d = load_rc2(args.rc2, 7.0)
    crop = (lambda a: centre(a, 64)) if args.dry else (lambda a: a)
    mask, chi, w = (crop(d["mask"]).astype(np.float32), crop(d["chi"]), crop(d["weight"]))
    voxel = d["voxel_size"]
    b = forward_field(chi, mask, voxel)
    starts = {"dirty": crop(d["field"]) * mask}
    if not args.dry:
        starts["noise_clean"] = nib.load("runs/s4b_noise_clean/best_clean.nii.gz").get_fdata(dtype=np.float32) * mask
    sets["sim2"] = dict(mask=mask, chi=chi, w=w, voxel=voxel, b=b, starts=starts, set=dict(te=8e-3, b0=7.0, pad=64 if args.dry else 256),
                        calc=(chi < -0.5))
    r = load_rc1(args.rc1)
    rc = (lambda a: centre(a, 64)) if args.dry else (lambda a: a)
    rmask, rchi, rw = rc(r["mask"]).astype(np.float32), rc(r["chi"]), rc(r["magnitude"])
    exps = rc1_experiments(r) if not args.dry else {}
    for name in ("exp7", "exp8", "exp9"):
        if args.dry:
            continue
        field = exps[name][0]
        sets[f"rc1_{name}"] = dict(mask=rmask, chi=rchi, w=rw, voxel=r["voxel_size"], b=forward_field(rchi, rmask, r["voxel_size"]),
                                   starts={"dirty": np.asarray(field, np.float32) * rmask}, set=dict(te=r["te"], b0=r["b0"], pad=192))
    return sets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rc1", default="dataRC1")
    parser.add_argument("--rc2", default="dataRC2")
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--dry", action="store_true", help="64^3 crop, 2 iterations: only checks that the script runs")
    parser.add_argument("--out", type=Path, default=Path("runs/gates"))
    args = parser.parse_args()
    if args.dry:
        args.iters = 2
    limit_vram()
    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = args.out / ("results_dry.json" if args.dry else "results.json")
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    sets = load_sets(args)

    for sname, s in sets.items():
        mask, chi, w, voxel, b = s["mask"], s["chi"], s["w"], s["voxel"], s["b"]
        shape = b.shape
        D = np.abs(dipole_kernel(shape, voxel, dtype=torch.float64).numpy())
        fr = np.meshgrid(*[np.fft.fftfreq(n, d=v) for n, v in zip(shape, voxel)], indexing="ij")
        K = np.sqrt(sum(f ** 2 for f in fr))
        B = np.fft.fftn(b)

        def score(key, field):
            full = f"{sname}/{key}"
            if full not in cache:
                cache[full] = invert(np.ascontiguousarray(field * mask, dtype=np.float64), mask, w, chi, args.iters, **s["set"])
                cache_path.write_text(json.dumps(cache))
                print(f"{full:46s} {cache[full]:7.2f}", flush=True)
            return cache[full]

        score("oracle", b)
        for start, f0 in s["starts"].items():
            score(start, f0)
            F0 = np.fft.fftn(f0)
            for d0 in NOTCH:
                score(f"{start}/notch{d0}", np.real(np.fft.ifftn(F0 * np.exp(-((d0 / np.maximum(D, 1e-9)) ** 4)))))
            for rname, rf in REGIONS.items():
                reg = rf(D, K)
                score(f"{start}/fix {rname}", np.real(np.fft.ifftn(np.where(reg, B, F0))))
            if sname == "sim2":
                dist = ndimage.distance_transform_edt(~s["calc"], sampling=voxel)
                for r in (3.0, 6.0):
                    sh = (dist <= r) & (mask > 0)
                    score(f"{start}/shell {r:g}mm", np.where(sh, b, f0))

    lines = [f"eNDI NRMSE % at {args.iters} fixed iterations; 'closure' = (start - result) / (start - oracle)"]
    for sname, s in sets.items():
        o = cache[f"{sname}/oracle"]
        lines.append(f"\n[{sname}] oracle {o:.2f}")
        for start in s["starts"]:
            st = cache[f"{sname}/{start}"]
            lines.append(f"  start = {start}: {st:.2f}")
            for key in sorted(k for k in cache if k.startswith(f"{sname}/{start}/")):
                v = cache[key]
                lines.append(f"    {key.split('/', 2)[2]:16s} {v:7.2f}   closure {100 * (st - v) / max(st - o, 1e-9):6.1f}%")
    text = "\n".join(lines)
    (args.out / ("table_dry.txt" if args.dry else "table.txt")).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
