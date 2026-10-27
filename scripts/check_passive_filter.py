"""Passive stage A (CLAUDE.md 9au): g1 refitted with every gain bounded to [0, 1] (the filter can only attenuate), same data as fit_regime_init.py
(STI+noise and off-diagonal+noise of RC1 at SNR 100, never the measured tissue residual). Stage A alone, eNDI@25, laptop CPU:
RC1 exp5 / 6 / 7 / 8 / 9 and the real Sim2. Run: python -I scripts/check_passive_filter.py"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import lsq_linear

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cascade_gate import DIP, G1, StageA  # noqa: E402
from rescore_fixed import invert  # noqa: E402

from mcno_qsm.artifacts import ArtifactConfig, Volume, make_dirty  # noqa: E402
from mcno_qsm.dipolelets import DipoleletTransform, image_components  # noqa: E402
from mcno_qsm.rc1 import clean_field, load_rc1  # noqa: E402
from mcno_qsm.rc2 import load_rc2  # noqa: E402

d = load_rc1("dataRC1")
mask, chi, vox = d["mask"].astype(np.float32), d["chi"], d["voxel_size"]
b = clean_field(chi, vox, mask)
vol = Volume.from_rc1(d, "cpu")
cfg = ArtifactConfig(snr_low=100.0, snr_high=100.0)
tr = DipoleletTransform(b.shape, vox, **DIP)
fits = []
for fam in ("sti_noise", "offdiag_noise"):
    dirty, _ = make_dirty(fam, vol, np.random.default_rng(3), torch.Generator().manual_seed(3), cfg)
    f = dirty.numpy() * mask
    cs = [p.numpy() * mask for p in image_components(tr, torch.from_numpy(np.ascontiguousarray(f, np.float32)))]
    A = np.stack([c[mask > 0] for c in cs], 1).astype(np.float64)
    fits.append(lsq_linear(A, b[mask > 0].astype(np.float64), bounds=(0.0, 1.0)).x)
g1 = np.mean(fits, axis=0)
print("unconstrained g1:", G1)
print("passive g1 [0,1]:", np.round(g1, 3).tolist(), flush=True)

at25 = lambda h: h["nrmse"][min(24, len(h["nrmse"]) - 1)]
c = np.load("runs/generalization/rc1_common.npz")
chi_g, mk, wt = c["chi_gt"].astype(np.float32), c["mask"].astype(np.float32), c["weight"].astype(np.float32)
for exp in ("exp5", "exp6", "exp7", "exp8", "exp9"):
    f = np.load(f"runs/generalization/rc1_{exp}.npz")["field_dirty"].astype(np.float32)
    out = StageA(vox, "cpu", None, g1)(f, mk, vox)[0]
    print(f"rc1 {exp} A passive eNDI@25 {at25(invert(out, chi_g, mk, wt, 25e-3, 2.8936, 192, 25)):7.2f}%", flush=True)
r2 = load_rc2("dataRC2", 7.0)
m = r2["mask"].astype(np.float32)
out = StageA(r2["voxel_size"], "cpu", None, g1)(r2["field"] * m, m, r2["voxel_size"])[0]
print(f"sim2 real A passive eNDI@25 {at25(invert(out.astype(np.float32), r2['chi'], m, r2['weight'], 8e-3, 7.0, 256, 25)):7.2f}%", flush=True)
