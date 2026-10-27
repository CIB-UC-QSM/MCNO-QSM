"""Scale robustness of PhaseCleaner runs (CLAUDE.md 9bd; server GPU or laptop, field domain, no eNDI, no training).

The nets were never trained with a varying weight scale (9u: a rescaled weight made them fail) nor with fields much stronger than RC1's. Normalized layers are meant to
make the blocks insensitive to such scales. Per run (default noise_clean and noise_clean_norm), on the forward-simulated Sim2 field `sim2_snrcal` (intact field + synthetic noise)
and on the real Sim2 field:
  weight x1 / x1.5 / x2.8 (x2.8 = the rescale that broke the nets in 9u, mean 0.71 instead of 0.25),
  amplitude of the CLEAN field x1 / x2.5 at fixed noise (field = a * clean + noise; truth = a * clean; snrcal only).
Reports rms(out - true) / rms(true) inside the mask and the ratio to the error at the reference condition (x1), plus the error of the dirty input.
Run (server): MCNO_VRAM_GB=10 .venv/bin/python scripts/test_scale_robustness.py --runs noise_clean noise_clean_norm > runs/scale_robustness.log 2>&1
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cascade_gate as cg  # noqa: E402

from mcno_qsm.rc2 import load_rc2  # noqa: E402
from mcno_qsm.scoring import forward_field, limit_vram  # noqa: E402

CONDITIONS = [("weight x1", 1.0, 1.0), ("weight x1.5", 1.5, 1.0), ("weight x2.8", 2.8, 1.0), ("amplitude x2.5", 1.0, 2.5), ("weight x2.8 + amplitude x2.5", 2.8, 2.5)]


def rel_error(out, true, mask):
    m = mask > 0
    return float(np.sqrt(np.sum((out[m] - true[m]) ** 2) / np.sum(true[m] ** 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["noise_clean", "noise_clean_norm"])
    parser.add_argument("--dir", type=Path, default=Path("runs/generalization_regime"))
    parser.add_argument("--rc2", default="dataRC2")
    parser.add_argument("--out", type=Path, default=Path("runs/scale_robustness"))
    parser.add_argument("--dry", action="store_true", help="laptop smoke test: untrained models, real Sim2 only, central 64^3 crop")
    args = parser.parse_args()
    limit_vram()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    models = {name: cg.load_b_run(name, device, args.dry) for name in args.runs}
    crop = (lambda a: a[tuple(slice((n - 64) // 2, (n - 64) // 2 + 64) for n in a.shape[:3])]) if args.dry else (lambda a: a)
    d2 = load_rc2(args.rc2, 7.0)
    voxel, mask = d2["voxel_size"], crop(d2["mask"].astype(np.float32))
    ref_real = crop(forward_field(d2["chi"], d2["mask"].astype(np.float32), voxel))
    sets = [("real Sim2", crop(d2["field"]) * mask, ref_real * mask, ref_real, crop(d2["weight"]).astype(np.float32), "out0", [c for c in CONDITIONS if c[2] == 1.0 and c[1] <= 2.8 and "+" not in c[0]])]
    if not args.dry:
        common = np.load(args.dir / "sim2_common.npz")
        path = next((args.dir / n for n in ("sim2_sim2_snrcal.npz", "sim2_snrcal.npz") if (args.dir / n).exists()))
        z = np.load(path)
        clean = common["clean"].astype(np.float32)
        noise = (z["field_dirty"].astype(np.float32) - clean) * mask
        w = (z["weight"] if "weight" in z.files else common["weight"]).astype(np.float32)
        sets.append(("sim2_snrcal (forward + noise)", clean * mask + noise, clean * mask, clean, w, "clean", CONDITIONS))
    lines = ["scale robustness: rms(out - true)/rms(true) in the mask  [ratio to the weight x1 / amplitude x1 condition]; dirty = the input itself"]
    for set_name, field, true, reference, weight, outside, conditions in sets:
        lines.append(f"== {set_name}")
        base = {}
        for label, w_scale, amp in conditions:
            clean_part = reference * mask
            noise_part = field - true if set_name.startswith("sim2_snrcal") else None
            f = (amp * clean_part + noise_part) if noise_part is not None else field
            t = amp * clean_part if noise_part is not None else true
            ref = reference * amp if noise_part is not None else reference
            cells = [f"dirty {rel_error(f * mask, t, mask):5.3f}"]
            for name, model in models.items():
                out = cg.stage_b(model, f, mask, weight * w_scale, ref, voxel, outside, device)
                e = rel_error(out, t, mask)
                base.setdefault(name, e)
                cells.append(f"{name} {e:5.3f} [{e / base[name]:4.2f}]")
            lines.append(f"{label:30s} " + "   ".join(cells))
    text = "\n".join(lines)
    print(text)
    (args.out / ("table_dry.txt" if args.dry else "table.txt")).write_text(text + "\n")


if __name__ == "__main__":
    main()
