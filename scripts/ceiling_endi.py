"""Ceiling of eNDI on PERFECT phase (CLAUDE.md 9bi), no training, eNDI only (ndiCG, alpha 1e-5, same inversion as every table).

What does eNDI give when the phase is exact, and how much does the brain mask cost? Fields (all the true forward field b = D*chi, correct kernel, unpadded):
  masked   mask * b                      the training target of the whole project (the 'oracle', 43.10 at 25 iterations on Sim2)
  dil1/2/3 b * (mask dilated by r voxels, 6-connected)   a rim of true field around the mask
  whole    b everywhere in the volume    no mask loss at all
The eNDI weight of a field that extends beyond the mask is the real weight inside the mask and the in-mask MEDIAN weight in the added voxels (zero outside the
region; the data weight beyond the mask does not exist in the data). The NRMSE is always measured inside the brain mask against the ground truth chi.
Each field is inverted for --iters iterations (default 150) and the whole curve is stored; the table reads @10, @25, @40, @100, @150 and the minimum.
Optional --runs: also the cleaned masked field of s4b runs (runs/s4b_<name>/best_clean.nii.gz) and the dirty field, to see whether THEY converge.
Sets: real Sim2 (TE 8 ms, 7 T, pad 256) and RC1 (25 ms, 2.89 T, pad 192).
Run (server): MCNO_VRAM_GB=10 nohup .venv/bin/python scripts/ceiling_endi.py --runs noise_clean > runs/ceiling.log 2>&1 &
Dry run (laptop): python -I scripts/ceiling_endi.py --dry   (64^3 crop, 3 iterations, no runs)
Writes runs/ceiling/meta.json (resumable) and runs/ceiling/table.txt."""

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rescore_fixed import invert  # noqa: E402

from mcno_qsm.rc1 import clean_field, load_rc1  # noqa: E402
from mcno_qsm.rc2 import load_rc2  # noqa: E402
from mcno_qsm.scoring import forward_field, limit_vram  # noqa: E402

SETTINGS = {"rc1": (25e-3, 2.8936, 192), "sim2": (8e-3, 7.0, 256)}
READ = (10, 25, 40, 100, 150)


def dilate(mask, radius):
    return binary_dilation(mask > 0, generate_binary_structure(3, 1), iterations=radius)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rc1", default="dataRC1")
    parser.add_argument("--rc2", default="dataRC2")
    parser.add_argument("--iters", type=int, default=150)
    parser.add_argument("--runs", nargs="*", default=[], help="s4b runs whose best_clean.nii.gz (real Sim2) is also inverted")
    parser.add_argument("--sets", nargs="*", default=["sim2", "rc1"])
    parser.add_argument("--out", type=Path, default=Path("runs/ceiling"))
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    if args.dry:
        args.iters, args.runs = 3, []
    limit_vram()
    args.out.mkdir(parents=True, exist_ok=True)
    meta_path = args.out / ("meta_dry.json" if args.dry else "meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    crop = (lambda a: a[tuple(slice((n - 64) // 2, (n - 64) // 2 + 64) for n in a.shape[:3])]) if args.dry else (lambda a: a)

    def run(tag, name, field, chi, mask, weight):
        key = f"{tag}/{name}"
        if key in meta:
            return
        te, b0, pad = SETTINGS[tag]
        meta[key] = invert(field.astype(np.float32), chi, mask, weight.astype(np.float32), te, b0, pad, args.iters)
        meta_path.write_text(json.dumps(meta))
        h = meta[key]["nrmse"]
        print(f"{tag:5s}{name:22s} " + " ".join(f"@{i}: {h[min(i, len(h) - 1)]:6.2f}" for i in READ) + f"  min {min(h):6.2f} @{h.index(min(h))}", flush=True)

    for tag in args.sets:
        if tag == "sim2":
            d = load_rc2(args.rc2, 7.0)
            chi, mask, weight, voxel = crop(d["chi"]), crop(d["mask"]).astype(np.float32), crop(d["weight"]), d["voxel_size"]
            real = crop(d["field"]) * mask
            cleaned = {r: crop(nib.load(Path("runs") / f"s4b_{r}" / "best_clean.nii.gz").get_fdata(dtype=np.float32)) * mask for r in args.runs}
        else:
            d = load_rc1(args.rc1)
            d = {k: (crop(v) if isinstance(v, np.ndarray) and v.ndim >= 3 else v) for k, v in d.items()}
            chi, mask, weight, voxel = d["chi"], d["mask"].astype(np.float32), d["magnitude"], d["voxel_size"]
            real, cleaned = None, {}
        b = forward_field(chi, None, voxel)
        inside = mask > 0
        med = float(np.median(weight[inside]))
        regions = {"masked": inside, **{f"dil{r}": dilate(mask, r) for r in (1, 2, 3)}, "whole": np.ones(mask.shape, bool)}
        e_in = float(np.sum(b[inside] ** 2))
        for name, region in regions.items():
            share = 100 * float(np.sum(b[region] ** 2)) / float(np.sum(b ** 2))
            print(f"{tag} {name}: {region.mean() * 100:5.1f}% of the volume, {share:5.1f}% of the field energy (mask alone {100 * e_in / float(np.sum(b ** 2)):.1f}%)", flush=True)
            w = np.where(region, np.where(inside, weight, med), 0.0)
            run(tag, name, b * region, chi, mask, w)
        if real is not None:
            run(tag, "dirty (real Frequency)", real, chi, mask, weight)
            for r, f in cleaned.items():
                run(tag, f"cleaned {r}", f, chi, mask, weight)

    rows = [f"eNDI NRMSE % inside the mask against chi (ndiCG, alpha 1e-5), by iteration; masked = the project's target",
            f"{'set / field':32s}" + "".join(f"{'@' + str(i):>9s}" for i in READ) + f"{'min':>9s}{'@min':>6s}"]
    for key in sorted(meta, key=lambda k: (k.split('/')[0], list(meta).index(k))):
        h = meta[key]["nrmse"]
        rows.append(f"{key:32s}" + "".join(f"{h[min(i, len(h) - 1)]:9.2f}" for i in READ) + f"{min(h):9.2f}{h.index(min(h)):6d}")
    text = "\n".join(rows)
    print(text)
    (args.out / ("table_dry.txt" if args.dry else "table.txt")).write_text(text + "\n")


if __name__ == "__main__":
    main()
