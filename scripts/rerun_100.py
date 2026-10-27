"""Earlier results re-run at a larger iteration count, and the dilated-mask (rim) inference option (CLAUDE.md 9bm). No training; eNDI only (ndiCG, alpha 1e-5).

Why: the perfect masked phase is still improving at 25 iterations (42.9 -> 36.7 at 150) and the cleaned fields behave differently from the dirty one beyond 25
(dirty diverges, 9bj); the 25-iteration protocol is an early stop. This re-runs the main earlier methods for 100 iterations and reads @25 / @40 / @100 and the minimum.
Methods per field: dirty | B (noise_clean) | A12 (the 12-number passive Dipole-let filter, no gate) | AG (gated passive filter) | AB (AG then B) | regime (joint net) | realmix_plain
(the last two come from the export files; the real Sim2 uses the best_clean volumes, so those need runs/s4b_<name>/best_clean.nii.gz).
Rim option (REAL Sim2 only, the one data set whose input has data in the voxels just outside the mask): the same methods applied with mask = brain mask dilated by R voxels
(default 2: the first two outside voxels hold data, 9bk), field = Frequency * dilated mask, eNDI weight * dilated mask; the NRMSE is still measured inside the brain mask.
Sets: RC1 exp1, exp3, exp7, exp8, exp9; forward Sim2 snrcal, snr40, jumps; real Sim2 (with and without rim).
Run (server): PYTORCH_ALLOC_CONF=expandable_segments:True MCNO_VRAM_GB=8 nohup .venv/bin/python scripts/rerun_100.py --cpu-b > runs/rerun100.log 2>&1 &
Dry run (laptop): python -I scripts/rerun_100.py --dry     Writes runs/rerun100/{meta.json,table.txt} (resumable)."""

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import binary_dilation, generate_binary_structure

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cascade_gate import G1_PASSIVE, SETTINGS, StageA, load_b_run, load_noise_clean, stage_b  # noqa: E402
from export_claim_views import a12  # noqa: E402
from rescore_fixed import invert  # noqa: E402

from mcno_qsm.rc2 import load_rc2  # noqa: E402
from mcno_qsm.scoring import forward_field, limit_vram  # noqa: E402

READ = (25, 40, 100)
STRUCT = generate_binary_structure(3, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("runs/generalization_regime"))
    parser.add_argument("--rc2", default="dataRC2")
    parser.add_argument("--rc1-exps", nargs="*", default=["exp1", "exp3", "exp7", "exp8", "exp9"])
    parser.add_argument("--sim2-exps", nargs="*", default=["sim2_snrcal", "sim2_snr40", "sim2_jumps"])
    parser.add_argument("--rim", type=int, default=2, help="dilation (voxels) of the rim variants of the real Sim2; 0 = none")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--cpu-b", action="store_true", help="run the networks and the stage-A filter on the CPU (no GPU memory for them)")
    parser.add_argument("--out", type=Path, default=Path("runs/rerun100"))
    parser.add_argument("--dry", action="store_true", help="laptop smoke test: 64^3 crop, untrained nets, 3 iterations, exp7 + real Sim2")
    args = parser.parse_args()
    if args.dry:
        args.iters, args.rc1_exps, args.sim2_exps, args.dir = 3, ["exp7"], [], Path("runs/generalization")
        SETTINGS.update({"rc1": (25e-3, 2.8936, 64), "sim2": (8e-3, 7.0, 64)})     # the 64^3 crop needs no 192^3 / 256^3 padding
    limit_vram()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_b = "cpu" if args.cpu_b else device
    args.out.mkdir(parents=True, exist_ok=True)
    meta_path = args.out / ("meta_dry.json" if args.dry else "meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    crop = (lambda a: a[tuple(slice((n - 64) // 2, (n - 64) // 2 + 64) for n in a.shape[:3])]) if args.dry else (lambda a: a)
    model_b = load_noise_clean(device_b, args.dry)
    model_regime = load_b_run("regime", device_b, args.dry)

    def run(tag, exp, name, field, chi, mask, weight):
        key = f"{tag}/{exp}/{name}"
        if key in meta:
            return
        te, b0, pad = SETTINGS[tag]
        meta[key] = invert(field.astype(np.float32), chi, mask, weight.astype(np.float32), te, b0, pad, args.iters)
        meta_path.write_text(json.dumps(meta))
        h = meta[key]["nrmse"]
        print(f"{tag:5s}{exp:12s}{name:20s} " + " ".join(f"@{i}: {h[min(i, len(h) - 1)]:6.2f}" for i in READ) + f"  min {min(h):6.2f} @{h.index(min(h))}", flush=True)
        torch.cuda.empty_cache()

    def methods(tag, exp, dirty, chi, mask, weight, reference, voxel, outside, exported, suffix="", region=None):
        """All methods on one field. `mask` is the region the networks work in (brain mask or its dilation); the eNDI weight is weight * region."""
        region = mask if region is None else region
        stage = StageA(voxel, device_b, None, G1_PASSIVE)
        w = weight * region
        brain = mask if suffix == "" else region_brain[0]
        run(tag, exp, "dirty" + suffix, dirty * region, chi, brain, w)
        run(tag, exp, "B" + suffix, stage_b(model_b, dirty, region, weight, reference, voxel, outside, device_b) * region, chi, brain, w)
        run(tag, exp, "A12" + suffix, a12(stage, dirty, region, voxel), chi, brain, w)
        ag = stage(dirty, region, voxel)[0]
        run(tag, exp, "AG" + suffix, ag, chi, brain, w)
        run(tag, exp, "AB" + suffix, stage_b(model_b, ag, region, weight, reference, voxel, outside, device_b) * region, chi, brain, w)
        if suffix == "":
            for m in ("regime", "realmix_plain"):
                if m in exported:
                    run(tag, exp, m, exported[m] * region, chi, brain, w)
        else:
            run(tag, exp, "regime" + suffix, stage_b(model_regime, dirty, region, weight, reference, voxel, outside, device_b) * region, chi, brain, w)

    region_brain = [None]
    for tag, exps in (("rc1", args.rc1_exps), ("sim2", args.sim2_exps)):
        if not exps or not (args.dir / f"{tag}_common.npz").exists():
            continue
        c = np.load(args.dir / f"{tag}_common.npz")
        chi, mask, ref, wt = (crop(c[k]).astype(np.float32) for k in ("chi_gt", "mask", "clean", "weight"))
        voxel = tuple(float(v) for v in c["voxel_size"])
        for exp in exps:
            z = np.load(args.dir / f"{tag}_{exp}.npz")
            weight = crop(z["weight"]).astype(np.float32) if "weight" in z.files else wt
            exported = {m: crop(z[f"field_{m}"]).astype(np.float32) * mask for m in ("regime", "realmix_plain") if f"field_{m}" in z.files}
            methods(tag, exp, crop(z["field_dirty"]).astype(np.float32), chi, mask, weight, ref, voxel, "clean", exported)

    # ---- real Sim2: without and with the rim
    d2 = load_rc2(args.rc2, 7.0)
    chi, mask, weight, voxel = crop(d2["chi"]), crop(d2["mask"]).astype(np.float32), crop(d2["weight"]).astype(np.float32), d2["voxel_size"]
    freq = crop(d2["field"]).astype(np.float32)
    reference = forward_field(chi, mask, voxel)
    exported = {}
    for m in ("regime", "realmix_plain"):
        p = Path("runs") / f"s4b_{m}" / "best_clean.nii.gz"
        if p.exists() and not args.dry:
            exported[m] = crop(nib.load(p).get_fdata(dtype=np.float32)) * mask
    methods("sim2", "sim2_real", freq * mask, chi, mask, weight, reference, voxel, "out0", exported)
    if args.rim > 0:
        dil = binary_dilation(mask > 0, STRUCT, iterations=args.rim).astype(np.float32)
        region_brain[0] = mask
        methods("sim2", "sim2_real", freq * dil, chi, dil, weight, forward_field(chi, dil, voxel), voxel, "out0", {}, suffix=f"_rim{args.rim}", region=dil)

    rows = [f"eNDI NRMSE % inside the brain mask vs chi (ndiCG alpha 1e-5), read @25 / @40 / @100 iterations and the minimum (iteration); rim variants = mask dilated by {args.rim} voxels",
            f"{'set / field':36s}" + "".join(f"{'@' + str(i):>9s}" for i in READ) + f"{'min':>9s}{'@min':>6s}"]
    for key in meta:
        h = meta[key]["nrmse"]
        rows.append(f"{key:36s}" + "".join(f"{h[min(i, len(h) - 1)]:9.2f}" for i in READ) + f"{min(h):9.2f}{h.index(min(h)):6d}")
    text = "\n".join(rows)
    print(text)
    (args.out / ("table_dry.txt" if args.dry else "table.txt")).write_text(text + "\n")


if __name__ == "__main__":
    main()
