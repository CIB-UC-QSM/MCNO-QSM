"""Passive stage A + strong-source suppressor read from RC1 (CLAUDE.md 9au): g1 bounded to [0,1] (values of check_passive_filter.py), gate x (1 - smoothstep((lrms - 0.06) / 0.04)),
lrms = local rms with Gaussian sigma 2.5 voxels. Stage A alone, eNDI@25, laptop CPU. Run: python -I scripts/check_passive_strong.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cascade_gate import StageA  # noqa: E402
from rescore_fixed import invert  # noqa: E402

from mcno_qsm.rc2 import load_rc2  # noqa: E402

G1P = [0.0, 0.0, 0.398, 0.0, 0.744, 1.0, 0.0, 0.773, 0.828, 0.0, 0.674, 0.919]
STRONG, SIGMA = (0.06, 0.10), 2.5
at25 = lambda h: h["nrmse"][min(24, len(h["nrmse"]) - 1)]
c = np.load("runs/generalization/rc1_common.npz")
chi, mk, wt = c["chi_gt"].astype(np.float32), c["mask"].astype(np.float32), c["weight"].astype(np.float32)
vox = tuple(float(v) for v in c["voxel_size"])
for exp in ("exp5", "exp6", "exp7", "exp8", "exp9"):
    f = np.load(f"runs/generalization/rc1_{exp}.npz")["field_dirty"].astype(np.float32)
    out, _, r = StageA(vox, "cpu", STRONG, G1P, SIGMA)(f, mk, vox)
    print(f"rc1 {exp} A passive+suppressor eNDI@25 {at25(invert(out, chi, mk, wt, 25e-3, 2.8936, 192, 25)):7.2f}%   r mean {r[mk > 0].mean():.2f}", flush=True)
d = load_rc2("dataRC2", 7.0)
m = d["mask"].astype(np.float32)
out, _, r = StageA(d["voxel_size"], "cpu", STRONG, G1P, SIGMA)(d["field"] * m, m, d["voxel_size"])
print(f"sim2 real A passive+suppressor eNDI@25 {at25(invert(out.astype(np.float32), d['chi'], m, d['weight'], 8e-3, 7.0, 256, 25)):7.2f}%   r mean {r[m > 0].mean():.2f}", flush=True)
