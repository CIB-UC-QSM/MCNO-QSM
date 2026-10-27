"""More radial scales instead of splitting the coarse remainder? (CLAUDE.md 9aq) Linear test, no training.
Dipole-let transform with J = 3, 4, 5, 6 detail scales x 3 |D| bands (+ coarse, 3J+1 segments), same Sim2 parameters otherwise, with and without the
coarse remainder split into 3 |D| bands. Best real gain per segment by least squares (dirty -> b, inside the mask): phase NRMSE % for
 fit      gains fitted on the same volume (optimistic)
 xfer     mean of the gains fitted on the three OTHER real types (RC1 tissue / STI / off-diag), applied to this one (Sim2 real: fitted on the 3 RC1 types)
 half     fitted on x < N/2, tested on x >= N/2
Sets: RC1 noise psi (SNR 60, synthetic), RC1 tissue / STI / off-diag (real residuals), Sim2 real (Frequency.nii)."""
import sys
from pathlib import Path

import nibabel as nib  # noqa: F401
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcno_qsm.artifacts import ArtifactConfig, Volume, make_dirty  # noqa: E402
from mcno_qsm.dipolelets import SIM2_DIPOLELET, DipoleletTransform, create_dipolelet_windows, image_components  # noqa: E402
from mcno_qsm.physics import dipole_kernel  # noqa: E402
from mcno_qsm.rc1 import clean_field, load_rc1, sti_fields  # noqa: E402
from mcno_qsm.rc2 import load_rc2  # noqa: E402
from mcno_qsm.scoring import forward_field  # noqa: E402

JS = (3, 4, 5, 6)
d = load_rc1("dataRC1")
mask1, chi1, vox1 = d["mask"].astype(np.float32), d["chi"], d["voxel_size"]
b1 = clean_field(chi1, vox1, mask1)
psti, pq = sti_fields(d["chi_13"], d["chi_23"], d["chi_33"], vox1)
vol = Volume.from_rc1(d, "cpu")
noisy, _ = make_dirty("noise", vol, np.random.default_rng(0), torch.Generator().manual_seed(0), ArtifactConfig(snr_low=60.0, snr_high=60.0))
r2 = load_rc2("dataRC2", 7.0)
mask2, vox2 = r2["mask"].astype(np.float32), r2["voxel_size"]
b2 = forward_field(r2["chi"], mask2, vox2)
SETS = {
    "noise psi": (noisy.numpy() * mask1, b1, mask1, vox1),
    "tissue": (mask1 * d["phs_tissue"], b1, mask1, vox1),
    "STI": (mask1 * psti, b1, mask1, vox1),
    "off-diag": (mask1 * (b1 - pq), b1, mask1, vox1),
    "Sim2 real": (r2["field"] * mask2, b2, mask2, vox2),
}
REAL = ("tissue", "STI", "off-diag")


def components(field, mask, vox, J, split):
    tr = DipoleletTransform(field.shape, vox, **{**SIM2_DIPOLELET, "scales": J})
    xt = torch.from_numpy(np.ascontiguousarray(field, dtype=np.float32))
    parts = [p.numpy() * mask for p in image_components(tr, xt)]
    if not split:
        return parts
    kern = dipole_kernel(field.shape, vox, dtype=torch.float32)
    thr = np.logspace(np.log10(SIM2_DIPOLELET["delta"]), np.log10(1 / 3), 2).tolist()
    win = create_dipolelet_windows(kern, thr, None, window="gaussian", width_factor=SIM2_DIPOLELET["gaussian_width_factor"])
    spec = list(tr.analysis(xt[None, None]))[-1][0, 0]
    sp = [torch.fft.ifftn(spec * w, dim=(-3, -2, -1), norm="ortho").real.numpy() * mask for w in win]
    return parts[:-1] + sp


def fit(cs, target, sel):
    A = np.stack([c[sel] for c in cs], 1).astype(np.float64)
    return np.linalg.lstsq(A, target[sel].astype(np.float64), rcond=None)[0]


def nrmse(cs, g, target, sel):
    pred = sum(gi * c for gi, c in zip(g, cs))
    return 100 * np.linalg.norm((pred - target)[sel]) / np.linalg.norm(target[sel])


rows = []
for J in JS:
    for split in (False, True):
        comp = {k: components(f, m, v, J, split) for k, (f, b, m, v) in SETS.items()}
        n = len(comp["noise psi"])
        gfit = {k: fit(comp[k], SETS[k][1], SETS[k][2] > 0) for k in SETS}
        out = {}
        for k, (f, b, m, v) in SETS.items():
            sel = m > 0
            others = [o for o in REAL if o != k]
            gx = np.mean([gfit[o] for o in others], axis=0) if k in REAL else np.mean([gfit[o] for o in REAL], axis=0)
            half = np.zeros_like(sel); half[: sel.shape[0] // 2] = True
            gh = fit(comp[k], b, sel & half)
            tm = sel & ~half
            out[k] = (nrmse(comp[k], gfit[k], b, sel), nrmse(comp[k], gx, b, sel) if k != "noise psi" else float("nan"), nrmse(comp[k], gh, b, tm))
        rows.append((J, split, n, out))
        print(f"J={J} split={split} segments={n}", flush=True)

dirty = {k: 100 * np.linalg.norm((f - b)[m > 0]) / np.linalg.norm(b[m > 0]) for k, (f, b, m, v) in SETS.items()}
print("\nphase NRMSE %  (dirty: " + ", ".join(f"{k} {v:.1f}" for k, v in dirty.items()) + ")")
for key, title in ((0, "FIT (same volume)"), (1, "XFER (gains from the other real types)"), (2, "HALF (fit x<N/2, test x>=N/2)")):
    print(f"\n{title}")
    print(f"{'decomposition':26s} " + " ".join(f"{k:>10s}" for k in SETS))
    for J, split, n, out in rows:
        name = f"{J} scales x3{' + coarse split' if split else ''} ({n})"
        print(f"{name:26s} " + " ".join(f"{out[k][key]:10.2f}" for k in SETS))
