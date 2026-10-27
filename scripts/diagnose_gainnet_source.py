"""Does the gain network keep an INTACT strong source? (CLAUDE.md 9bb; no training, no eNDI, RC1 head + Sim2 forward field)

The weight counterfactual (diagnose_gainnet_weight.py) showed that the erasure of the calcification field on the forward-simulated Sim2 does not come from the
weight channel. Here (1) strong sources like the training family `calc_shell` (|chi| 0.8-2.0 ppm, radius 2-3.5 voxels, ppm of the same size as Sim2's calcification)
are added to the RC1 head in two ways: INTACT (field untouched, notebook noise at SNR 100, weight untouched) and CORRUPTED (the training recipe: shell with only 10-70 %
of the field plus noise, weight dip), plus noise only as the control; the trained network (stage A' alone, GNa; B behind it, GN) is applied to each and the field removed
in the neighbourhood of the sources is measured. (2) On Sim2's sim2_snrcal the channels `amp` (log local rms), `phi` (local cone excess) and both are replaced by
their in-mask median (GainNet.neutralize) to see which input drives the erasure. RC1 draws are on a TRAINING head (in-sample head, fresh draws); this is a behaviour check.
Regions on RC1: |source field| > 0.05 ppm ('near'), 0.01-0.05 ('ring'), < 0.01 ('far'). Columns: rms(out - true)/rms(true) / % of the true field removed.
Run (server): MCNO_VRAM_GB=10 .venv/bin/python scripts/diagnose_gainnet_source.py > runs/diagnose_gainnet_source.log 2>&1      Laptop smoke test: python -I scripts/diagnose_gainnet_source.py --dry
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

from mcno_qsm.artifacts import ArtifactConfig, Volume, make_dirty  # noqa: E402
from mcno_qsm.rc1 import load_rc1  # noqa: E402
from mcno_qsm.rc2 import load_rc2  # noqa: E402
from mcno_qsm.scoring import forward_field, limit_vram  # noqa: E402

CONDITIONS = {
    "noise only (no source)": ("noise", {}),
    "INTACT strong source": ("calc_shell", dict(calc_keep=(1.0, 1.0), calc_noise=(0.0, 0.0))),
    "CORRUPTED shell (training recipe)": ("calc_shell", {}),
}


class Pool:
    """Pooled sums over draws: error energy, true energy and the projection of the removed field on the truth."""

    def __init__(self):
        self.sums = {}

    def add(self, region_name, method, out, dirty, true, region):
        e = self.sums.setdefault((region_name, method), np.zeros(4))
        e += [float(np.sum((out[region] - true[region]) ** 2)), float(np.sum(true[region] ** 2)),
              float(np.sum((dirty[region] - out[region]) * true[region])), float(region.sum())]

    def cell(self, region_name, method):
        e = self.sums[(region_name, method)]
        return f"{np.sqrt(e[0] / e[1]):5.3f}/{100 * e[2] / e[1]:6.1f}%"


def rc1_part(args, device, gain_model, model_b, lines):
    d = load_rc1(args.rc1)
    v = Volume.from_rc1(d, "cpu")
    mask = d["mask"].astype(np.float32)
    voxel = d["voxel_size"]
    stage_a = cg.StageA(voxel, device, None, cg.G1_PASSIVE)
    for name, (family, overrides) in CONDITIONS.items():
        cfg = ArtifactConfig(snr_low=100.0, snr_high=100.0, **overrides)
        pool = Pool()
        for seed in range(args.draws):
            rng, gen = np.random.default_rng(100 + seed), torch.Generator().manual_seed(100 + seed)
            dirty_t, info = make_dirty(family, v, rng, gen, cfg)
            clean = info["clean"] if "clean" in info else v.clean
            weight = (info["magnitude"] if "magnitude" in info else v.magnitude).numpy().astype(np.float32)
            dirty, clean_np = dirty_t.numpy().astype(np.float32), clean.numpy().astype(np.float32)
            true = clean_np * mask
            reference = clean_np
            soft, _, r = stage_a(dirty, mask, voxel)
            outs = {"dirty": dirty * mask, "A": soft, "B": cg.stage_b(model_b, dirty, mask, weight, reference, voxel, "clean", device)}
            outs["GNa"], outs["GN"] = cg.gain_outputs(gain_model, dirty, mask, weight, reference, voxel, "clean", device)
            source = np.abs(clean_np - v.clean.numpy())
            reg = {"near": (source > 0.05) & (mask > 0), "ring": (source > 0.01) & (source <= 0.05) & (mask > 0), "far": (source <= 0.01) & (mask > 0)}
            for rname, region in reg.items():
                if region.sum() == 0:
                    continue
                for method, out in outs.items():
                    pool.add(rname, method, out, outs["dirty"], true, region)
        lines.append(f"== RC1 head, {name}: {args.draws} draws")
        for rname in ("near", "ring", "far"):
            if (rname, "dirty") in pool.sums:
                n = int(pool.sums[(rname, "dirty")][3])
                lines.append(f"{rname:5s} n={n:9d}  " + "  ".join(f"{m}: {pool.cell(rname, m)}" for m in ("dirty", "A", "GNa", "B", "GN")))


def sim2_part(args, device, gain_model, lines):
    d2 = load_rc2(args.rc2, 7.0)
    voxel = d2["voxel_size"]
    mask, chi = d2["mask"].astype(np.float32), d2["chi"]
    common = np.load(args.dir / "sim2_common.npz")
    path = next((args.dir / n for n in ("sim2_sim2_snrcal.npz", "sim2_snrcal.npz") if (args.dir / n).exists()), None)
    if path is None:
        lines.append("== sim2_snrcal: file not found")
        return
    z = np.load(path)
    w = (z["weight"] if "weight" in z.files else common["weight"]).astype(np.float32)
    field, clean = z["field_dirty"].astype(np.float32) * mask, common["clean"].astype(np.float32)
    stage_a = cg.StageA(voxel, device, None, cg.G1_PASSIVE)
    reg = regions(mask, chi, voxel, stage_a(field, mask, voxel)[2])
    lines.append("== Sim2 sim2_snrcal (forward simulation, intact field), GNa with input channels neutralized (in-mask median)")
    for label, neutral in (("as is", ()), ("amp neutral", ("amp",)), ("phi neutral", ("phi",)), ("amp+phi neutral", ("amp", "phi")), ("weight neutral", ("weight",))):
        gain_model.gain.neutralize = neutral
        out = cg.gain_outputs(gain_model, field, mask, w, clean, voxel, "clean", device)[0]
        cells = []
        for rname in ("<=3mm", "3-8mm", "8-15mm"):
            region, true = reg[rname], clean * mask
            err = np.sqrt(np.sum((out[region] - true[region]) ** 2) / np.sum(true[region] ** 2))
            along = 100 * np.sum((field[region] - out[region]) * true[region]) / np.sum(true[region] ** 2)
            cells.append(f"{rname}: {err:5.3f}/{along:6.1f}%")
        lines.append(f"{label:16s} " + "   ".join(cells))
    gain_model.gain.neutralize = ()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="gainnet")
    parser.add_argument("--rc1", default="dataRC1")
    parser.add_argument("--rc2", default="dataRC2")
    parser.add_argument("--dir", type=Path, default=Path("runs/generalization_regime"))
    parser.add_argument("--draws", type=int, default=4)
    parser.add_argument("--out", type=Path, default=Path("runs/diagnose_gainnet"))
    parser.add_argument("--dry", action="store_true", help="laptop smoke test: untrained networks, one draw, RC1 part only")
    args = parser.parse_args()
    limit_vram()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    model_b = cg.load_noise_clean(device, args.dry)
    gain_model = cg.load_gain_run(args.run, device, args.dry, model_b)
    lines = ["does the gain network keep a strong source? error / % of the true field removed (see the script header)"]
    if args.dry:
        args.draws = 1                       # laptop smoke test: untrained networks, one draw, the Sim2 part needs the server exports
    rc1_part(args, device, gain_model, model_b, lines)
    if not args.dry:
        sim2_part(args, device, gain_model, lines)
    text = "\n".join(lines)
    print(text)
    (args.out / ("source_dry.txt" if args.dry else "source.txt")).write_text(text + "\n")


if __name__ == "__main__":
    main()
