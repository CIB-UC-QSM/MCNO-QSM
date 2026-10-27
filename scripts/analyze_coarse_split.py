"""Does splitting the coarse remainder by |D| help a LINEAR per-segment filter on real residuals? (CLAUDE.md 9ap)
Best real gain per segment by least squares (dirty -> b, inside the mask), for RC1 psi types; 10 segments (Sim2 Dipole-let) vs
10 with the coarse split in 3 |D| bands (same thresholds) vs radial scales only (4 segments) vs a hard cone notch.
Gains are fitted on the same volume (optimistic; 4-12 parameters); a transfer column fits on the noise psi and applies to the real one."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcno_qsm.artifacts import ArtifactConfig, Volume, make_dirty  # noqa: E402
from mcno_qsm.dipolelets import SIM2_DIPOLELET, DipoleletTransform, create_dipolelet_windows, image_components  # noqa: E402
from mcno_qsm.physics import dipole_kernel  # noqa: E402
from mcno_qsm.rc1 import clean_field, load_rc1, sti_fields  # noqa: E402

d = load_rc1("dataRC1")
mask, chi, voxel = d["mask"].astype(np.float32), d["chi"], d["voxel_size"]
shape = chi.shape
b = clean_field(chi, voxel, mask)
psti, pq = sti_fields(d["chi_13"], d["chi_23"], d["chi_33"], voxel)
vol = Volume.from_rc1(d, "cpu")
noisy, _ = make_dirty("noise", vol, np.random.default_rng(0), torch.Generator().manual_seed(0), ArtifactConfig(snr_low=60.0, snr_high=60.0))
dirty = {"noise SNR60": noisy.numpy() * mask, "tissue": mask * d["phs_tissue"], "STI": mask * psti, "off-diag": mask * (b + (-pq))}
tr = DipoleletTransform(shape, voxel, **SIM2_DIPOLELET)
kernel = dipole_kernel(shape, voxel, dtype=torch.float32)
delta = SIM2_DIPOLELET["delta"]
thr = np.logspace(np.log10(delta), np.log10(1 / 3), 2).tolist()
win = create_dipolelet_windows(kernel, thr, None, window="gaussian", width_factor=SIM2_DIPOLELET["gaussian_width_factor"])  # [3,X,Y,Z]
print("coarse split windows (mass of k-space):", [round(float(w.mean()), 3) for w in win])

def comps(x):
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    c10 = [p.numpy() * mask for p in image_components(tr, xt)]
    spec = list(tr.analysis(xt[None, None]))[-1][0, 0]
    split = [torch.fft.ifftn(spec * w, dim=(-3, -2, -1), norm="ortho").real.numpy() * mask for w in win]
    radial = [sum(c10[3 * j:3 * j + 3]) for j in range(3)] + [c10[9]]
    return {"10 seg": c10, "12 seg (coarse split)": c10[:9] + split, "radial 4": radial}

m = mask > 0
def fit(cs, target):
    A = np.stack([c[m] for c in cs], 1).astype(np.float64)
    g, *_ = np.linalg.lstsq(A, target[m].astype(np.float64), rcond=None)
    return g
def nrmse(cs, g, target):
    pred = sum(gi * c for gi, c in zip(g, cs))
    return 100 * np.linalg.norm((pred - target)[m]) / np.linalg.norm(target[m])
bm = b
res = {k: comps(v) for k, v in dirty.items()}
gnoise = {name: fit(cs, bm) for name, cs in res["noise SNR60"].items()}
print(f"\n{'psi type':14s} {'dirty':>7s} | " + " | ".join(f"{n:>22s}" for n in res['noise SNR60']) + " | cone notch(0.05) | cone notch(0.1)")
for k, v in dirty.items():
    row = f"{k:14s} {100*np.linalg.norm((v-bm)[m])/np.linalg.norm(bm[m]):7.2f} | "
    cells = []
    for name, cs in res[k].items():
        g = fit(cs, bm)
        cells.append(f"fit {nrmse(cs, g, bm):6.2f} xfer {nrmse(cs, gnoise[name], bm):6.2f}")
    F = np.fft.fftn(v)
    D = np.abs(kernel.numpy())
    notch = [100 * np.linalg.norm(((np.real(np.fft.ifftn(F * np.exp(-((t / np.maximum(D, 1e-9)) ** 4)))) * mask) - bm)[m]) / np.linalg.norm(bm[m]) for t in (0.05, 0.1)]
    print(row + " | ".join(cells) + f" | {notch[0]:6.2f} | {notch[1]:6.2f}")
for k in dirty:
    if k != "noise SNR60":
        print(f"\ngains fitted on {k} (12 seg, coarse split last 3):", np.round(fit(res[k]['12 seg (coarse split)'], bm), 3))

# honesty checks: (1) fit on one real psi type, apply to the others; (2) fit on the first half of the volume (x < 80), test on x >= 80
print("transfer between real psi types, 12 seg (coarse split) vs 10 seg: rows = fitted on, cols = applied to (phase NRMSE %)")
types = ["tissue", "STI", "off-diag"]
for name in ("10 seg", "12 seg (coarse split)"):
    print(name)
    for a in types:
        g = fit(res[a][name], bm)
        print(f"  fit {a:9s} -> " + "  ".join(f"{b_}: {nrmse(res[b_][name], g, bm):6.2f}" for b_ in types))
    gall = np.mean([fit(res[a][name], bm) for a in types], axis=0)
    print("  mean of the three fits -> " + "  ".join(f"{b_}: {nrmse(res[b_][name], gall, bm):6.2f}" for b_ in types))
half = np.zeros_like(m); half[:80] = True
print("half-volume check (fit x<80, test x>=80), phase NRMSE % on the test half")
def fit_h(cs, target, sel):
    A = np.stack([c[sel] for c in cs], 1).astype(np.float64)
    return np.linalg.lstsq(A, target[sel].astype(np.float64), rcond=None)[0]
tm = m & ~half
for k in types:
    out = []
    for name in ("10 seg", "12 seg (coarse split)"):
        g = fit_h(res[k][name], bm, m & half)
        pred = sum(gi * c for gi, c in zip(g, res[k][name]))
        out.append(f"{name}: {100*np.linalg.norm((pred-bm)[tm])/np.linalg.norm(bm[tm]):6.2f}")
    print(f"{k:9s} dirty {100*np.linalg.norm((dirty[k]-bm)[tm])/np.linalg.norm(bm[tm]):6.2f}  " + "  ".join(out))
