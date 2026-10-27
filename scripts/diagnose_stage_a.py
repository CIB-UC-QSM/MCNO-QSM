"""Where does stage A of the cascade (CLAUDE.md 9as) act and damage on the REAL Sim2? CPU, no network.
Prints the regime map r by region, and the field error (vs mask*D*chi) of the dirty field and of stage A inside / outside r > 0.5 and by distance to the
calcification (Chi < -0.5 ppm). Run: python -I scripts/diagnose_stage_a.py"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cascade_gate import StageA  # noqa: E402

from mcno_qsm.rc2 import load_rc2  # noqa: E402
from mcno_qsm.scoring import forward_field  # noqa: E402

d = load_rc2("dataRC2", 7.0)
m = d["mask"].astype(np.float32)
vox = d["voxel_size"]
tgt = forward_field(d["chi"], m, vox)
dirty = d["field"] * m
soft, hard, r = StageA(vox, "cpu")(dirty, m, vox)
mk = m > 0
dist = distance_transform_edt(d["chi"] >= -0.5) * np.mean(vox)
err = lambda f, s: float(np.sqrt(np.sum(((f - tgt)[s]) ** 2)))
print(f"voxels in mask {mk.sum()}, r > 0.5: {(mk & (r > 0.5)).sum()} ({100 * (mk & (r > 0.5)).sum() / mk.sum():.1f}%), r > 0.2: {(mk & (r > 0.2)).sum()}")
print(f"{'region':26s}{'voxels':>9s}{'|dirty-tgt|':>13s}{'|A-tgt|':>10s}{'ratio A/dirty':>15s}{'rms tgt':>10s}")
regions = {"r > 0.5": mk & (r > 0.5), "r 0.2-0.5": mk & (r > 0.2) & (r <= 0.5), "r <= 0.2": mk & (r <= 0.2),
           "<= 3 mm of calcification": mk & (dist <= 3), "3-8 mm": mk & (dist > 3) & (dist <= 8), "8-15 mm": mk & (dist > 8) & (dist <= 15),
           "> 15 mm": mk & (dist > 15), "r > 0.5 and > 15 mm": mk & (r > 0.5) & (dist > 15)}
for name, s in regions.items():
    if s.sum():
        a, b = err(dirty, s), err(soft, s)
        print(f"{name:26s}{s.sum():9d}{a:13.4f}{b:10.4f}{b / max(a, 1e-12):15.2f}{np.sqrt(np.mean(tgt[s] ** 2)):10.4f}")
tot = lambda f: float(np.sqrt(np.sum((f - tgt)[mk] ** 2)))
print(f"whole mask: |dirty-tgt| {tot(dirty):.3f}  |A-tgt| {tot(soft):.3f}  (phase NRMSE dirty {100 * tot(dirty) / np.linalg.norm(tgt[mk]):.1f}%, A {100 * tot(soft) / np.linalg.norm(tgt[mk]):.1f}%)")
share = lambda s: 100 * float(np.sum(((soft - tgt)[s]) ** 2) - np.sum(((dirty - tgt)[s]) ** 2)) / max(float(np.sum(((soft - tgt)[mk]) ** 2) - np.sum(((dirty - tgt)[mk]) ** 2)), 1e-12)
print("share of the net change in squared error (A - dirty) by region:", {k: round(share(s), 1) for k, s in regions.items() if s.sum()})
