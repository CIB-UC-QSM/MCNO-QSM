"""Rim probe on the real Sim2 (CLAUDE.md 9bk): is there anything to gain from a cleaned field that extends a few voxels beyond the brain mask?
No training; the EXISTING noise_clean network and eNDI only (ndiCG, alpha 1e-5).

Sim2's Frequency.nii holds data in the first two voxels outside the mask (relative error 0.8 and 1.07 vs the true field, against 0.42 inside; the third voxel is 94% zeros, 9bk).
Fields, all inverted with weight = the real eNDI weight times the region indicator and scored inside the BRAIN mask against chi:
  dirty_in           Frequency * mask                                              (weight * mask)
  clean_in           noise_clean(Frequency * mask)                                 (weight * mask)       the 47.31 row
  clean_in+dirty_rN  clean_in inside the mask, raw Frequency in the rim of N voxels (weight on mask dilated by N)   the old 'dirty outside' protocol
  clean_rim_rN       noise_clean applied with mask = mask dilated by N (mask channel, field and loss region), field = Frequency * dilated mask
  oracle_masked / oracle_dilN / oracle_whole   the TRUE field b = D*chi on the same regions (weight as in ceiling_endi.py): the ceiling of each region
For every field: NRMSE inside the mask @25 and @100 iterations, the calcification ROI (Chi < -0.5 ppm, dilated 8 voxels), a streak ring (8-25 voxels from those voxels,
inside the mask) and the rest of the mask; plus the sagittal plane through the calcification of the chi map (float16) for the notebook.
Run (server): MCNO_VRAM_GB=10 nohup .venv/bin/python scripts/rim_probe.py > runs/rim_probe.log 2>&1 &
Dry run (laptop, 64^3 crop, untrained B, 3 iterations): python -I scripts/rim_probe.py --dry
Writes runs/rim_probe/{meta.json,planes.npz,table.txt} (resumable)."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import binary_dilation, distance_transform_edt, generate_binary_structure

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cascade_gate import load_noise_clean, stage_b  # noqa: E402
from export_claim_views import endi_chi  # noqa: E402

from mcno_qsm.rc2 import load_rc2  # noqa: E402
from mcno_qsm.scoring import forward_field, limit_vram  # noqa: E402

STRUCT = generate_binary_structure(3, 1)
READ = (25, 100)


def nrmse(x, chi, region):
    return float(100 * np.linalg.norm((x - chi)[region]) / np.linalg.norm(chi[region]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rc2", default="dataRC2")
    parser.add_argument("--radii", nargs="*", type=int, default=[1, 2, 3])
    parser.add_argument("--out", type=Path, default=Path("runs/rim_probe"))
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--cpu-b", action="store_true", help="run the noise_clean network on the CPU (slow, no GPU memory) so only the eNDI inversions use the shared GPU")
    args = parser.parse_args()
    iters = (2, 3) if args.dry else READ
    limit_vram()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    meta_path = args.out / ("meta_dry.json" if args.dry else "meta.json")
    planes_path = args.out / ("planes_dry.npz" if args.dry else "planes.npz")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    planes = {k: v for k, v in np.load(planes_path).items() if k != "centre"} if planes_path.exists() else {}
    crop = (lambda a: a[tuple(slice((n - 64) // 2, (n - 64) // 2 + 64) for n in a.shape[:3])]) if args.dry else (lambda a: a)

    d = load_rc2(args.rc2, 7.0)
    chi, mask, weight, voxel = crop(d["chi"]), crop(d["mask"]).astype(np.float32), crop(d["weight"]).astype(np.float32), d["voxel_size"]
    freq = crop(d["field"]).astype(np.float32)
    inside = mask > 0
    b = forward_field(chi, None, voxel)
    device_b = "cpu" if args.cpu_b else device
    model = load_noise_clean(device_b, args.dry)
    dil = {0: inside, **{r: binary_dilation(inside, STRUCT, iterations=r) for r in args.radii}}
    # ROI / ring around the strong sources
    strong = chi < -0.5
    if strong.any():
        dist = distance_transform_edt(~strong, sampling=voxel)
        roi, ring = inside & (dist <= 8 * voxel[0]), inside & (dist > 8 * voxel[0]) & (dist <= 25 * voxel[0])
        centre = tuple(int(v) for v in np.unravel_index(np.argmin(chi), chi.shape))
    else:
        roi = ring = np.zeros_like(inside)
        centre = tuple(n // 2 for n in chi.shape)
    rest = inside & ~roi & ~ring
    print(f"calcification voxels {int(strong.sum())}, ROI {int(roi.sum())}, ring {int(ring.sum())}, plane x = {centre[0]}", flush=True)

    med = float(np.median(weight[inside]))
    fields = {}
    wr = lambda r: weight * dil[r]
    fields["dirty_in"] = (freq * mask, wr(0))
    clean_in = stage_b(model, freq, mask, weight, forward_field(chi, mask, voxel), voxel, "out0", device_b)
    torch.cuda.empty_cache()
    fields["clean_in"] = (clean_in * mask, wr(0))
    for r in args.radii:
        rim = (dil[r] & ~inside).astype(np.float32)
        fields[f"clean_in+dirty_r{r}"] = (clean_in * mask + freq * rim, wr(r))
        mr = dil[r].astype(np.float32)
        out = stage_b(model, freq * mr, mr, weight * mr, forward_field(chi, mr, voxel), voxel, "out0", device_b)
        torch.cuda.empty_cache()
        fields[f"clean_rim_r{r}"] = (out * mr, wr(r))
    fields["oracle_masked"] = (b * inside, wr(0))
    for r in args.radii:
        fields[f"oracle_dil{r}"] = (b * dil[r], np.where(dil[r], np.where(inside, weight, med), 0.0).astype(np.float32))
    fields["oracle_whole"] = (b, np.where(inside, weight, med).astype(np.float32))

    for name, (field, w) in fields.items():
        for n in iters:
            key = f"{name}/{n}"
            if key in meta:
                continue
            torch.cuda.empty_cache()
            x, hist = endi_chi(field.astype(np.float32), chi, mask, w.astype(np.float32), "sim2", n)
            meta[key] = {"global": nrmse(x, chi, inside), "roi": nrmse(x, chi, roi) if roi.any() else None,
                         "ring": nrmse(x, chi, ring) if ring.any() else None, "rest": nrmse(x, chi, rest) if rest.any() else None}
            planes[f"{name}/{n}"] = (x * mask)[centre[0]].astype(np.float16)
            meta_path.write_text(json.dumps(meta))
            np.savez_compressed(planes_path, **planes, centre=np.array(centre))
            m = meta[key]
            print(f"{name:22s} @{n:3d}: global {m['global']:6.2f}  ROI {m['roi'] or float('nan'):6.2f}  ring {m['ring'] or float('nan'):6.2f}  rest {m['rest'] or float('nan'):6.2f}", flush=True)
    planes["gt"] = (chi * mask)[centre[0]].astype(np.float16)
    np.savez_compressed(planes_path, **planes, centre=np.array(centre))

    rows = [f"eNDI NRMSE % inside the brain mask vs chi (ndiCG alpha 1e-5); ROI = Chi < -0.5 dilated 8 voxels, ring = 8-25 voxels from the strong voxels, rest = other mask voxels",
            f"{'field':22s}" + "".join(f"{k + '@' + str(n):>12s}" for n in iters for k in ('global', 'ROI', 'ring', 'rest'))]
    for name in fields:
        cells = []
        for n in iters:
            m = meta.get(f"{name}/{n}", {})
            cells += [f"{m[k]:12.2f}" if m.get(k) is not None else f"{'-':>12s}" for k in ("global", "roi", "ring", "rest")]
        rows.append(f"{name:22s}" + "".join(cells))
    text = "\n".join(rows)
    print(text)
    (args.out / ("table_dry.txt" if args.dry else "table.txt")).write_text(text + "\n")


if __name__ == "__main__":
    main()
