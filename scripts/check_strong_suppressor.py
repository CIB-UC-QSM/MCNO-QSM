"""Stage A alone with the strong-source suppressor (CLAUDE.md 9at), eNDI@25 on the laptop CPU: RC1 exp5, exp6, exp7 and the real Sim2.
Thresholds fixed from RC1 before this run: lo 0.03, hi 0.05 ppm. Run: python -I scripts/check_strong_suppressor.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cascade_gate import StageA  # noqa: E402
from rescore_fixed import invert  # noqa: E402

from mcno_qsm.rc2 import load_rc2  # noqa: E402

STRONG = (0.03, 0.05)
at25 = lambda h: h["nrmse"][min(24, len(h["nrmse"]) - 1)]
c = np.load("runs/generalization/rc1_common.npz")
chi, mk, wt = c["chi_gt"].astype(np.float32), c["mask"].astype(np.float32), c["weight"].astype(np.float32)
vox = tuple(float(v) for v in c["voxel_size"])
for exp in ("exp5", "exp6", "exp7"):
    f = np.load(f"runs/generalization/rc1_{exp}.npz")["field_dirty"].astype(np.float32)
    for name, strong in (("A", None), ("A+suppressor", STRONG)):
        out, _, r = StageA(vox, "cpu", strong)(f, mk, vox)
        h = invert(out, chi, mk, wt, 25e-3, 2.8936, 192, 25)
        print(f"rc1 {exp} {name:13s} eNDI@25 {at25(h):7.2f}%   r mean {r[mk > 0].mean():.2f}", flush=True)
    print(f"rc1 {exp} dirty         eNDI@25 {at25(invert(f, chi, mk, wt, 25e-3, 2.8936, 192, 25)):7.2f}%", flush=True)
d = load_rc2("dataRC2", 7.0)
m = d["mask"].astype(np.float32)
dirty = d["field"] * m
for name, strong in (("dirty", "dirty"), ("A", None), ("A+suppressor", STRONG)):
    f = dirty if strong == "dirty" else StageA(d["voxel_size"], "cpu", strong)(dirty, m, d["voxel_size"])[0]
    print(f"sim2 real {name:13s} eNDI@25 {at25(invert(f.astype(np.float32), d['chi'], m, d['weight'], 8e-3, 7.0, 256, 25)):7.2f}%", flush=True)
