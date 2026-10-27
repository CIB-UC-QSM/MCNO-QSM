"""Can the regime (noise-like psi vs real in-vivo-like psi) be read from the INPUT field alone? (CLAUDE.md 9ar) No truth is used for the feature.
Feature per Dipole-let scale j (3 scales + coarse): cone excess  R_j = (E_cone,j / E_other,j) / (mass_cone,j / mass_other,j), where E are the
Parseval energies of the 3 |D| bands (cone = band 0, |D| < 0.05) of the segment spectrum of the INPUT, and mass = the same ratio for a white-noise probe
(so R = 1 for white noise and for any field whose cone content is as strong as its non-cone content per unit k-space mass).
Computed on whole volumes and on random 96^3 patches inside the mask (the unit the network sees). Sets: clean b (reference), RC1 noise psi
(SNR 40 / 60 / 100 / 200), RC1 real residuals (tissue, STI, off-diag, as exp7-9), real Sim2 Frequency, the forward-simulated Sim2 + noise."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcno_qsm.artifacts import ArtifactConfig, Volume, make_dirty  # noqa: E402
from mcno_qsm.dipolelets import SIM2_DIPOLELET, DipoleletTransform  # noqa: E402
from mcno_qsm.rc1 import clean_field, load_rc1, sti_fields  # noqa: E402
from mcno_qsm.rc2 import load_rc2  # noqa: E402
from mcno_qsm.scoring import forward_field  # noqa: E402

PATCH = 96
rng = np.random.default_rng(0)
d = load_rc1("dataRC1")
mask1, chi1, vox1 = d["mask"].astype(np.float32), d["chi"], d["voxel_size"]
b1 = clean_field(chi1, vox1, mask1)
psti, pq = sti_fields(d["chi_13"], d["chi_23"], d["chi_33"], vox1)
vol = Volume.from_rc1(d, "cpu")
sets = {"clean b (RC1)": b1}
for snr in (40, 60, 100, 200):
    n, _ = make_dirty("noise", vol, np.random.default_rng(1), torch.Generator().manual_seed(1), ArtifactConfig(snr_low=snr, snr_high=snr))
    sets[f"noise SNR{snr}"] = n.numpy() * mask1
sets.update({"tissue (real) exp7": mask1 * d["phs_tissue"], "STI exp8": mask1 * psti, "off-diag exp9": mask1 * (b1 - pq)})
masks = {k: mask1 for k in sets}
r2 = load_rc2("dataRC2", 7.0)
m2 = r2["mask"].astype(np.float32)
b2 = forward_field(r2["chi"], m2, r2["voxel_size"])
sets["Sim2 real"] = r2["field"] * m2
sets["Sim2 clean b"] = b2
masks["Sim2 real"] = masks["Sim2 clean b"] = m2
vox = {k: (vox1 if "Sim2" not in k else r2["voxel_size"]) for k in sets}

transforms = {}


def transform(shape, v):
    key = (shape, v)
    if key not in transforms:
        transforms[key] = DipoleletTransform(shape, v, **SIM2_DIPOLELET)
    return transforms[key]


def features(x, shape, v):
    tr = transform(shape, v)
    e = np.array([float((s[0, 0].abs() ** 2).sum()) for s in tr.analysis(torch.from_numpy(np.ascontiguousarray(x, np.float32))[None, None])])
    return e


def ratios(x, shape, v, mass):
    e = features(x, shape, v)
    out = []
    for j in range(3):
        cone, other = e[3 * j], e[3 * j + 1] + e[3 * j + 2]
        out.append((cone / max(other, 1e-30)) / mass[j])
    return out


def mass_ratios(shape, v):
    e = features(np.random.default_rng(5).standard_normal(shape).astype(np.float32), shape, v)
    return [e[3 * j] / (e[3 * j + 1] + e[3 * j + 2]) for j in range(3)]


print(f"{'set':22s} | whole volume R1 R2 R3        | 96^3 patches inside the mask: median [min, max] of R1 / R2 / R3")
for k, x in sets.items():
    m = masks[k]
    shape = x.shape
    mass_whole = mass_ratios(shape, vox[k])
    whole = ratios(x, shape, vox[k], mass_whole)
    # patch statistics
    pshape = (PATCH,) * 3
    mass_patch = mass_ratios(pshape, vox[k])
    vals = []
    tries = 0
    while len(vals) < 12 and tries < 200:
        tries += 1
        o = [int(rng.integers(0, s - PATCH + 1)) for s in shape]
        sl = tuple(slice(i, i + PATCH) for i in o)
        if m[sl].mean() < 0.4:
            continue
        vals.append(ratios(x[sl] * m[sl], pshape, vox[k], mass_patch))
    vals = np.array(vals)
    cells = " ".join(f"{np.median(vals[:, j]):6.2f} [{vals[:, j].min():5.2f},{vals[:, j].max():5.2f}]" for j in range(3))
    print(f"{k:22s} | {whole[0]:6.2f} {whole[1]:6.2f} {whole[2]:6.2f}      | {cells}", flush=True)
