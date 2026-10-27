"""Where do the learned gains damage Sim2? (CLAUDE.md 9ba; server GPU or laptop CPU, no training, no eNDI.)

For the real Sim2 field and the forward-simulated Sim2 field `sim2_snrcal` (same head and calcification, the true residual is pure synthetic noise, so any
removal of true field is damage) it compares, per region, the fixed passive gated filter (stage A of the cascade) with the gain network (GNa) and, behind
the frozen noise_clean, AB and GN. Regions: distance to the calcification (Chi < -0.5 ppm voxels; <= 3, 3-8, 8-15, > 15 mm) and, farther than 15 mm, regime map r > 0.5 / <= 0.5.
Columns per region and method: error = rms(out - true) / rms(true) in the region (true = mask * D*Chi; lower is better, dirty is the reference),
along truth = sum((dirty - out) * true) / sum(true^2) in % (positive = the method removes true field), and the mean removal fraction 1 - g_s of the fixed filter and of the
gain network for the segments (segment order scale1/band0..2, scale2/band0..2, scale3/band0..2, coarse/band0..2).
Run (server): MCNO_VRAM_GB=10 .venv/bin/python scripts/diagnose_gainnet.py > runs/diagnose_gainnet.log 2>&1      Laptop smoke test: python -I scripts/diagnose_gainnet.py --dry
Writes runs/diagnose_gainnet/table.txt (and table_dry.txt)."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cascade_gate as cg  # noqa: E402

from mcno_qsm.rc2 import load_rc2  # noqa: E402
from mcno_qsm.scoring import forward_field, limit_vram  # noqa: E402

SEGMENTS = ["s1b0", "s1b1", "s1b2", "s2b0", "s2b1", "s2b2", "s3b0", "s3b1", "s3b2", "cob0", "cob1", "cob2"]


def regions(mask, chi, voxel, r):
    dist = distance_transform_edt(chi >= -0.5, sampling=voxel) if (chi < -0.5).any() else np.full(chi.shape, np.inf)
    m = mask > 0
    out = {"<=3mm": m & (dist <= 3), "3-8mm": m & (dist > 3) & (dist <= 8), "8-15mm": m & (dist > 8) & (dist <= 15), ">15mm": m & (dist > 15)}
    out[">15mm r>0.5"] = out[">15mm"] & (r > 0.5)
    out[">15mm r<=0.5"] = out[">15mm"] & (r <= 0.5)
    out["whole mask"] = m
    return out


def report(name, field, true_field, chi, mask, weight, reference, voxel, outside, device, stage_a, gain_model, model_b, lines):
    soft, _, r = stage_a(field, mask, voxel)
    outs = {"dirty": field * mask, "A": soft, "B": cg.stage_b(model_b, field, mask, weight, reference, voxel, outside, device)}
    outs["AB"] = cg.stage_b(model_b, soft, mask, weight, reference, voxel, outside, device)
    gain_model.gain.keep_gain = True
    outs["GNa"], outs["GN"] = cg.gain_outputs(gain_model, field, mask, weight, reference, voxel, outside, device)
    gains = gain_model.gain.last_gain[0].cpu().numpy()          # [S, X, Y, Z]
    removal_a = r[None] * (1 - np.asarray(cg.G1_PASSIVE, dtype=np.float32))[:, None, None, None]    # fixed filter: 1 - g_s = r (1 - g1_s)
    lines.append(f"== {name}")
    for label, region in regions(mask, chi, voxel, r).items():
        n = int(region.sum())
        if n < 50:
            continue
        true = true_field[region]
        rms = float(np.sqrt(np.mean(true ** 2)))
        cells = []
        for method in ("dirty", "A", "GNa", "B", "AB", "GN"):
            err = float(np.sqrt(np.mean((outs[method][region] - true) ** 2))) / rms
            along = 100 * float(np.sum((outs["dirty"][region] - outs[method][region]) * true) / np.sum(true ** 2))
            cells.append(f"{err:6.3f}/{along:6.1f}" if method != "dirty" else f"{err:6.3f}      ")
        lines.append(f"{label:13s} n={n:8d} rms(true)={rms:.4f}  error/along%: dirty {cells[0]}  A {cells[1]}  GNa {cells[2]}  B {cells[3]}  AB {cells[4]}  GN {cells[5]}")
        lines.append(f"{'':13s} mean removal 1-g: A  " + " ".join(f"{removal_a[s][region].mean():5.2f}" for s in range(len(SEGMENTS))))
        lines.append(f"{'':13s}                   GNa " + " ".join(f"{(1 - gains[s])[region].mean():5.2f}" for s in range(len(SEGMENTS))))
    lines.append("   segments:            " + " ".join(f"{s:>5s}" for s in SEGMENTS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="gainnet")
    parser.add_argument("--dir", type=Path, default=Path("runs/generalization_regime"))
    parser.add_argument("--rc2", default="dataRC2")
    parser.add_argument("--out", type=Path, default=Path("runs/diagnose_gainnet"))
    parser.add_argument("--dry", action="store_true", help="laptop smoke test: untrained networks, central 64^3 crop, real Sim2 only")
    args = parser.parse_args()
    limit_vram()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    model_b = cg.load_noise_clean(device, args.dry)
    gain_model = cg.load_gain_run(args.run, device, args.dry, model_b)
    crop = (lambda a: a[tuple(slice((n - 64) // 2, (n - 64) // 2 + 64) for n in a.shape[:3])]) if args.dry else (lambda a: a)
    d2 = load_rc2(args.rc2, 7.0)
    voxel = d2["voxel_size"]
    mask = crop(d2["mask"].astype(np.float32))
    chi = crop(d2["chi"])
    stage_a = cg.StageA(voxel, device, None, cg.G1_PASSIVE)
    lines = [f"diagnosis of the learned gains on Sim2 (run {args.run}); error = rms(out - true)/rms(true), along = % of the true field removed (see the script header)"]
    clean = crop(forward_field(d2["chi"], d2["mask"].astype(np.float32), voxel)) * mask
    reference = crop(forward_field(d2["chi"], d2["mask"].astype(np.float32), voxel))
    report("sim2_real (real Frequency.nii, zero outside the mask)", crop(d2["field"]) * mask, clean, chi, mask, crop(d2["weight"]), reference, voxel, "out0",
           device, stage_a, gain_model, model_b, lines)
    snr = args.dir / "sim2_snrcal.npz"
    if not args.dry and snr.exists():
        common = np.load(args.dir / "sim2_common.npz")
        z = np.load(snr)
        w = z["weight"].astype(np.float32) if "weight" in z.files else common["weight"].astype(np.float32)
        report("sim2_snrcal (forward simulation + synthetic noise only)", z["field_dirty"].astype(np.float32) * mask, common["clean"].astype(np.float32) * mask, chi,
               common["mask"].astype(np.float32), w, common["clean"].astype(np.float32), tuple(float(v) for v in common["voxel_size"]), "clean",
               device, stage_a, gain_model, model_b, lines)
    text = "\n".join(lines)
    print(text)
    (args.out / ("table_dry.txt" if args.dry else "table.txt")).write_text(text + "\n")


if __name__ == "__main__":
    main()
