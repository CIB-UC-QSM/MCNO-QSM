"""eNDI (25 fixed iterations) of the real Sim2 with stage A applied only in part of the volume (CPU; ~5 min per inversion). CLAUDE.md 9as.
Variants: dirty | A soft everywhere | A only > 15 mm from the calcification | A only <= 15 mm | A only where r > 0.5. Run: python -I scripts/diagnose_stage_a_endi.py"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cascade_gate import StageA  # noqa: E402
from rescore_fixed import invert  # noqa: E402

from mcno_qsm.rc2 import load_rc2  # noqa: E402

d = load_rc2("dataRC2", 7.0)
m = d["mask"].astype(np.float32)
vox = d["voxel_size"]
dirty = d["field"] * m
soft, _, r = StageA(vox, "cpu")(dirty, m, vox)
corr = soft - dirty                                     # = -r * sum_s (1 - g1_s) comp_s, masked
dist = distance_transform_edt(d["chi"] >= -0.5) * np.mean(vox)
variants = {"dirty": dirty, "A everywhere": soft, "A only > 15 mm from calc": dirty + corr * (dist > 15), "A only <= 15 mm": dirty + corr * (dist <= 15),
            "A only where r > 0.5": dirty + corr * (r > 0.5)}
for name, f in variants.items():
    t = time.time()
    h = invert(f.astype(np.float32), d["chi"], m, d["weight"], 8e-3, 7.0, 256, 25)
    print(f"{name:28s} eNDI@25 {h['nrmse'][min(24, len(h['nrmse']) - 1)]:6.2f}%   ({time.time() - t:.0f}s)", flush=True)
