"""Counterfactual for CLAUDE.md 9bb: is the weight dip around the calcification what makes the gain network erase the field? (no training, no eNDI)

The same fields are given to the trained gain network (GNa = stage A' alone) with the weight channel (a) as it is and (b) replaced inside the mask by its in-mask
median (no dip, no spatial structure); every other input is identical. Per region (distance to the calcification) it prints rms(out - true)/rms(true) and
'along' = % of the true field removed (see diagnose_gainnet.py). Sets: real Sim2 and the forward-simulated sim2_snrcal / sim2_jumps (intact field + synthetic noise).
Run (server): MCNO_VRAM_GB=10 .venv/bin/python scripts/diagnose_gainnet_weight.py > runs/diagnose_gainnet_weight.log 2>&1      Laptop smoke test: python -I scripts/diagnose_gainnet_weight.py --dry
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cascade_gate as cg  # noqa: E402
from diagnose_gainnet import regions  # noqa: E402

from mcno_qsm.rc2 import load_rc2  # noqa: E402
from mcno_qsm.scoring import forward_field, limit_vram  # noqa: E402


def run_set(name, field, true_field, chi, mask, weight, reference, voxel, outside, device, gain_model, lines):
    stage_a = cg.StageA(voxel, device, None, cg.G1_PASSIVE)
    r = stage_a(field, mask, voxel)[2]
    flat = np.where(mask > 0, np.median(weight[mask > 0]), 0).astype(np.float32)
    outs = {"dirty": field * mask}
    for label, w in (("weight as is", weight), ("weight flat", flat)):
        outs[label] = cg.gain_outputs(gain_model, field, mask, w, reference, voxel, outside, device)[0]
    lines.append(f"== {name}   (weight in the mask: median {np.median(weight[mask > 0]):.3f}, p1 {np.percentile(weight[mask > 0], 1):.3f}, p99 {np.percentile(weight[mask > 0], 99):.3f})")
    for label, region in regions(mask, chi, voxel, r).items():
        if int(region.sum()) < 50 or label.startswith(">15mm r"):
            continue
        true = true_field[region]
        rms = float(np.sqrt(np.mean(true ** 2)))
        cells = []
        for method in outs:
            err = float(np.sqrt(np.mean((outs[method][region] - true) ** 2))) / rms
            along = 100 * float(np.sum((outs["dirty"][region] - outs[method][region]) * true) / np.sum(true ** 2))
            cells.append(f"{method}: {err:5.3f}" + ("" if method == "dirty" else f"/{along:6.1f}%"))
        lines.append(f"{label:11s} n={int(region.sum()):8d}  " + "   ".join(cells))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="gainnet")
    parser.add_argument("--dir", type=Path, default=Path("runs/generalization_regime"))
    parser.add_argument("--rc2", default="dataRC2")
    parser.add_argument("--out", type=Path, default=Path("runs/diagnose_gainnet"))
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    limit_vram()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    model_b = cg.load_noise_clean(device, args.dry)
    gain_model = cg.load_gain_run(args.run, device, args.dry, model_b)
    crop = (lambda a: a[tuple(slice((n - 64) // 2, (n - 64) // 2 + 64) for n in a.shape[:3])]) if args.dry else (lambda a: a)
    d2 = load_rc2(args.rc2, 7.0)
    voxel = d2["voxel_size"]
    mask, chi = crop(d2["mask"].astype(np.float32)), crop(d2["chi"])
    ref = crop(forward_field(d2["chi"], d2["mask"].astype(np.float32), voxel))
    lines = ["counterfactual weight channel for the gain network alone (GNa); error / % of the true field removed"]
    run_set("sim2_real", crop(d2["field"]) * mask, ref * mask, chi, mask, crop(d2["weight"]).astype(np.float32), ref, voxel, "out0", device, gain_model, lines)
    if not args.dry:
        common = np.load(args.dir / "sim2_common.npz")
        for exp in ("sim2_snrcal", "sim2_jumps"):
            path = next((args.dir / n for n in (f"sim2_{exp}.npz", f"{exp}.npz") if (args.dir / n).exists()), None)
            if path is None:
                lines.append(f"== {exp}: file not found")
                continue
            z = np.load(path)
            w = (z["weight"] if "weight" in z.files else common["weight"]).astype(np.float32)
            run_set(exp, z["field_dirty"].astype(np.float32) * mask, common["clean"].astype(np.float32) * mask, chi, common["mask"].astype(np.float32), w,
                    common["clean"].astype(np.float32), tuple(float(v) for v in common["voxel_size"]), "clean", device, gain_model, lines)
    text = "\n".join(lines)
    print(text)
    (args.out / ("weight_dry.txt" if args.dry else "weight.txt")).write_text(text + "\n")


if __name__ == "__main__":
    main()
