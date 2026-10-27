"""Where does the REAL in-vivo-like residual sit in k-space? (CLAUDE.md 9ap) RC1 (the only head with a measured non-dipolar phase):
psi_tissue = mask*(phs_tissue - D chi_cosmos), psi_sti = mask*(pSTI - D chi_cosmos), psi_offdiag = -mask*pQ, psi_noise = synthetic SNR-60 draw
(for reference). Per Dipole-let segment energy shares, cone fractions, and local SNR |b|^2/|psi|^2 per |k| row and |D| column."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcno_qsm.artifacts import ArtifactConfig, Volume, make_dirty  # noqa: E402
from mcno_qsm.dipolelets import SIM2_DIPOLELET, DipoleletTransform  # noqa: E402
from mcno_qsm.physics import dipole_kernel  # noqa: E402
from mcno_qsm.rc1 import clean_field, load_rc1, sti_fields  # noqa: E402

d = load_rc1("dataRC1")
mask, chi, voxel = d["mask"].astype(np.float32), d["chi"], d["voxel_size"]
shape = chi.shape
b = clean_field(chi, voxel, mask)
psti, pq = sti_fields(d["chi_13"], d["chi_23"], d["chi_33"], voxel)
vol = Volume.from_rc1(d, "cpu")
cfg = ArtifactConfig(snr_low=60.0, snr_high=60.0)
noisy, _ = make_dirty("noise", vol, np.random.default_rng(0), torch.Generator().manual_seed(0), cfg)
psi = {"noise SNR60": noisy.numpy() * mask - b, "tissue (real)": mask * d["phs_tissue"] - b, "STI": mask * psti - b, "off-diag": -mask * pq}
names = [f"s{j}/b{k}" for j in (1, 2, 3) for k in (0, 1, 2)] + ["coarse"]
tr = DipoleletTransform(shape, voxel, **SIM2_DIPOLELET)
def sh(x):
    e = np.array([float((s[0, 0].abs() ** 2).sum()) for s in tr.analysis(torch.from_numpy(x.astype(np.float32))[None, None])])
    return 100 * e / e.sum()
print(f"{'signal':14s} rms(ppm) " + " ".join(f"{n:>6s}" for n in names))
print(f"{'b (true)':14s} {np.sqrt((b[mask>0]**2).mean()):8.4f} " + " ".join(f"{v:6.2f}" for v in sh(b)))
for k, v in psi.items():
    print(f"{k:14s} {np.sqrt((v[mask>0]**2).mean()):8.4f} " + " ".join(f"{x:6.2f}" for x in sh(v)))
adk = np.abs(dipole_kernel(shape, voxel, dtype=torch.float64).numpy())
fr = np.meshgrid(*[np.fft.fftfreq(n, d=v) for n, v in zip(shape, voxel)], indexing="ij")  # cycles/mm
kmag = np.sqrt(sum(f ** 2 for f in fr))
F = {"b": np.abs(np.fft.fftn(b)) ** 2, **{k: np.abs(np.fft.fftn(v)) ** 2 for k, v in psi.items()}}
print("\nfraction of energy with |D|<0.05 / <0.1, and with |k|<0.1 cyc/mm / <0.2:")
for k, a in F.items():
    print(f"{k:14s} |D|<.05 {100*a[adk<0.05].sum()/a.sum():6.2f}%  |D|<.1 {100*a[adk<0.1].sum()/a.sum():6.2f}%  |k|<.1 {100*a[kmag<0.1].sum()/a.sum():6.2f}%  |k|<.2 {100*a[kmag<0.2].sum()/a.sum():6.2f}%")
ke = np.array([0, 0.03, 0.06, 0.1, 0.15, 0.2, 0.3, 0.4, 1.0]); de = np.array([0, 0.05, 0.1, 0.2, 0.333, 0.7])
for k in psi:
    print(f"\n10log10(|b|^2/|psi|^2) for psi = {k}; rows |k| cyc/mm {ke[1:]}, cols |D| upper {de[1:]}")
    for i in range(len(ke) - 1):
        row = []
        for j in range(len(de) - 1):
            s = (kmag >= ke[i]) & (kmag < ke[i + 1]) & (adk >= de[j]) & (adk < de[j + 1])
            row.append(10 * np.log10(max(F["b"][s].sum(), 1e-30) / max(F[k][s].sum(), 1e-30)))
        print(f"{ke[i]:4.2f}-{ke[i+1]:4.2f} " + " ".join(f"{x:7.1f}" for x in row))
