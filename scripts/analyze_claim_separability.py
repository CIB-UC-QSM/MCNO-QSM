"""Evidence for claim 2 of CLAUDE.md 9bg: noise is separable from the dipole field by SCALE (|k|), the real non-dipolar residual by ANGLE (|D|).
RC1 (the only head with a measured non-dipolar phase). Residuals psi: synthetic noise (SNR 60, notebook recipe), measured tissue (phs_tissue - D chi_cosmos),
modelled STI, modelled off-diagonal. CPU, about a minute. Writes runs/claim_views/separability.npz.

For each psi, with b = mask * D chi_cosmos and d = mask * (b + psi) (the dirty input):
 (1) local SNR 10 log10(|B|^2 / |PSI|^2) per (|k|, |D|) cell;
 (2) the best scalar linear gain per cell g = <B, D>/<D, D> (least squares, fitted on the volume itself: an optimistic ORACLE) as a function of |D| (marginal over |k|)
     and of |k| (marginal over |D|);
 (3) phase NRMSE (%, mask) of the filtered dirty field when the gain may depend only on |k| (RADIAL), only on |D| (ANGULAR) or on both (CELL), with gains fitted on
     the same psi (same-regime) and applied to the other regimes (cross-regime): the cost of using the wrong geometry.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcno_qsm.artifacts import ArtifactConfig, Volume, make_dirty  # noqa: E402
from mcno_qsm.physics import dipole_kernel  # noqa: E402
from mcno_qsm.rc1 import clean_field, load_rc1, sti_fields  # noqa: E402

KE = np.array([0, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 1.0])           # |k| edges, cycles/mm
DE = np.array([0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.27, 0.333, 0.5, 0.7])       # |D| edges

d = load_rc1("dataRC1")
mask, chi, voxel = d["mask"].astype(np.float32), d["chi"], d["voxel_size"]
shape = chi.shape
b = clean_field(chi, voxel, mask)
psti, pq = sti_fields(d["chi_13"], d["chi_23"], d["chi_33"], voxel)
noisy, _ = make_dirty("noise", Volume.from_rc1(d, "cpu"), np.random.default_rng(0), torch.Generator().manual_seed(0),
                      ArtifactConfig(snr_low=60.0, snr_high=60.0))
psi = {"noise (SNR 60)": noisy.numpy() * mask - b, "tissue (measured)": mask * d["phs_tissue"] - b, "STI (modelled)": mask * psti - b,
       "off-diagonal (modelled)": -mask * pq}
adk = np.abs(dipole_kernel(shape, voxel, dtype=torch.float64).numpy())
fr = np.meshgrid(*[np.fft.fftfreq(n, d=v) for n, v in zip(shape, voxel)], indexing="ij")
kmag = np.sqrt(sum(f ** 2 for f in fr))
ik = np.clip(np.digitize(kmag, KE) - 1, 0, len(KE) - 2).ravel()
idd = np.clip(np.digitize(adk, DE) - 1, 0, len(DE) - 2).ravel()
nk, nd = len(KE) - 1, len(DE) - 1
B = np.fft.fftn(b).ravel()


def acc(values, flat, size):
    return np.bincount(flat, weights=values, minlength=size)


def gains(D, mode):
    """Least-squares scalar gain per cell of the chosen kind -> array in the shape of the grid used by `mode`."""
    flat, size = {"radial": (ik, nk), "angular": (idd, nd), "cell": (ik * nd + idd, nk * nd)}[mode]
    num, den = acc(np.real(B * np.conj(D)), flat, size), acc(np.abs(D) ** 2, flat, size)
    return num / np.maximum(den, 1e-30)


def apply(g, D, mode):
    flat = {"radial": ik, "angular": idd, "cell": ik * nd + idd}[mode]
    return np.real(np.fft.ifftn((g[flat] * D).reshape(shape)))


def nrmse(x):
    return float(100 * np.linalg.norm((x - b)[mask > 0]) / np.linalg.norm(b[mask > 0]))


out = {"KE": KE, "DE": DE}
dirty = {k: mask * (b + v) for k, v in psi.items()}
Dft = {k: np.fft.fftn(v).ravel() for k, v in dirty.items()}
Pft = {k: np.fft.fftn(v).ravel() for k, v in psi.items()}
flat = ik * nd + idd
Eb = acc(np.abs(B) ** 2, flat, nk * nd).reshape(nk, nd)
for name in psi:
    Ep = acc(np.abs(Pft[name]) ** 2, flat, nk * nd).reshape(nk, nd)
    out[f"snr_db__{name}"] = 10 * np.log10(np.maximum(Eb, 1e-30) / np.maximum(Ep, 1e-30))
    g = gains(Dft[name], "cell").reshape(nk, nd)
    out[f"gain_cell__{name}"] = g
    out[f"gain_radial__{name}"] = gains(Dft[name], "radial")
    out[f"gain_angular__{name}"] = gains(Dft[name], "angular")
    out[f"energy_cell__{name}"] = Ep
    out[f"nrmse_identity__{name}"] = nrmse(dirty[name])
out["energy_cell__b"] = Eb
names = list(psi)
for mode in ("radial", "angular", "cell"):
    m = np.zeros((len(names), len(names)))
    for i, fit in enumerate(names):
        g = gains(Dft[fit], mode)
        for j, test in enumerate(names):
            m[i, j] = nrmse(mask * apply(g, Dft[test], mode))
    out[f"cross__{mode}"] = m
out["names"] = np.array(names)
Path("runs/claim_views").mkdir(parents=True, exist_ok=True)
np.savez_compressed("runs/claim_views/separability.npz", **out)
print("phase NRMSE %, rows = regime the gains were fitted on, columns = regime they are applied to")
print(f"{'':26s}" + "".join(f"{n[:18]:>20s}" for n in names))
print(f"{'identity (dirty)':26s}" + "".join(f"{out[f'nrmse_identity__{n}']:20.2f}" for n in names))
for mode in ("radial", "angular", "cell"):
    for i, fit in enumerate(names):
        print(f"{mode + ' <- ' + fit[:12]:26s}" + "".join(f"{out[f'cross__{mode}'][i, j]:20.2f}" for j in range(len(names))))
