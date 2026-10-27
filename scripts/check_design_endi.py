"""eNDI validation of the data-derived Dipole-let decompositions (CLAUDE.md 9ax), laptop CPU, no network. Passive filters (gains in [0,1]) fitted on RC1's
STI+noise and off-diagonal+noise (SNR 100, as design_dipolelet2.py; the measured tissue is never fitted), applied UNGATED to the RC1 experiment fields exp5-exp9 and
scored with eNDI at 25 fixed iterations (history index 25). Designs: the current 12-segment Dipole-let and the real-regime designs of runs/dipolelet2/designs.json (S = 6, 8, 12).
Run: python -I scripts/check_design_endi.py   (needs runs/dipolelet2/designs.json and runs/generalization/rc1_exp*.npz)"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import lsq_linear

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from design_dipolelet2 import axes, comps_hard, windows  # noqa: E402
from rescore_fixed import invert  # noqa: E402

from mcno_qsm.artifacts import ArtifactConfig, Volume, make_dirty  # noqa: E402
from mcno_qsm.dipolelets import SIM2_DIPOLELET, DipoleletTransform, image_components  # noqa: E402
from mcno_qsm.rc1 import clean_field, load_rc1  # noqa: E402

d1 = load_rc1("dataRC1")
mask, chi, vox = d1["mask"].astype(np.float32), d1["chi"], d1["voxel_size"]
b = clean_field(chi, vox, mask)
mk = mask > 0
vol = Volume.from_rc1(d1, "cpu")
fit = {}
for fam in ("sti_noise", "offdiag_noise"):
    dirty, _ = make_dirty(fam, vol, np.random.default_rng(3), torch.Generator().manual_seed(3), ArtifactConfig(snr_low=100.0, snr_high=100.0))
    fit[fam] = dirty.numpy() * mask
kern, kmag = axes(b.shape, vox)
designs = json.load(open("runs/dipolelet2/designs.json"))["real"]
tr = DipoleletTransform(b.shape, vox, **{**SIM2_DIPOLELET, "coarse_bands": 3})
banks = {"current 12": lambda f: [p.numpy() * mask for p in image_components(tr, torch.from_numpy(np.ascontiguousarray(f, np.float32)))]}
for S in ("6", "8", "12"):
    groups = [(u, t, edges) for u, t, edges in designs[S]["groups"]]
    ws = windows(groups, kern, kmag)
    banks[f"derived S={S}"] = lambda f, ws=ws: comps_hard(ws, f, mask)

c = np.load("runs/generalization/rc1_common.npz")
chi_g, wt = c["chi_gt"].astype(np.float32), c["weight"].astype(np.float32)
exps = {e: np.load(f"runs/generalization/rc1_{e}.npz")["field_dirty"].astype(np.float32) for e in ("exp7", "exp8", "exp9", "exp5", "exp6")}
score = lambda f: invert(f.astype(np.float32), chi_g, mask, wt, 25e-3, 2.8936, 192, 25)["nrmse"][25]
rows = {"dirty": {e: score(f) for e, f in exps.items()}}
print("dirty", {e: round(v, 2) for e, v in rows["dirty"].items()}, flush=True)
for name, bank in banks.items():
    cf = [bank(fit[k]) for k in fit]
    A = np.concatenate([np.stack([x[mk] for x in cs], 1) for cs in cf]).astype(np.float64)
    y = np.concatenate([b[mk] for _ in fit]).astype(np.float64)
    g = lsq_linear(A, y, bounds=(0.0, 1.0)).x
    rows[name] = {e: score(sum(gi * x for gi, x in zip(g, bank(f)))) for e, f in exps.items()}
    print(name, "gains", np.round(g, 2).tolist(), {e: round(v, 2) for e, v in rows[name].items()}, flush=True)
print(f"\n{'eNDI@25 %, ungated':20s}" + "".join(f"{e:>9s}" for e in exps))
for name, r in rows.items():
    print(f"{name:20s}" + "".join(f"{r[e]:9.2f}" for e in exps))
