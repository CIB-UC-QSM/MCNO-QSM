"""Is there long-range streaking from Sim2's calcification? (CLAUDE.md 9ap) Excess chi error (method - oracle eNDI map) versus distance to the
37 calcification voxels: rms per voxel, and the rms of its near-cone (|D|<0.1) and far (|D|>=0.1) Fourier parts, in the mask."""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcno_qsm.physics import dipole_kernel  # noqa: E402
from mcno_qsm.rc2 import load_rc2  # noqa: E402

data = load_rc2("dataRC2", 7.0)
mask, chi, voxel = data["mask"] > 0, data["chi"], data["voxel_size"]
z = np.load("runs/chi_fixed25.npz")
c = {k[4:]: z[k].astype(np.float32) * mask for k in z.files}
adk = np.abs(dipole_kernel(chi.shape, voxel, dtype=torch.float64).numpy())
dist = ndimage.distance_transform_edt(~(chi < -0.5), sampling=voxel)
edges = [0, 3, 8, 15, 25, 40, 60, 200]
def parts(e):
    E = np.fft.fftn(e)
    near = np.real(np.fft.ifftn(E * (adk < 0.1))) * mask
    return near, e - near
print("rms (x1e-3 ppm) per voxel in distance shells; columns: oracle err | dirty-oracle diff total (near-cone / far) | noise_clean-oracle diff total (near / far)")
dd, dn = parts(c["dirty"] - c["oracle"]), parts(c["noise_clean"] - c["oracle"])
eo = parts(c["oracle"] - chi * mask)
print(f"{'dist mm':>9s} {'nvox':>8s} {'chi rms':>8s} | {'oracle err':>10s} (near/far) | {'dirty-orc':>9s} ({'near':>5s}/{'far':>5s}) | {'nc-orc':>7s} ({'near':>5s}/{'far':>5s})")
for lo, hi in zip(edges[:-1], edges[1:]):
    s = mask & (dist >= lo) & (dist < hi)
    r = lambda x: 1e3 * float(np.sqrt((x[s] ** 2).mean()))
    print(f"{lo:3d}-{hi:<5d} {int(s.sum()):8d} {r(chi):8.2f} | {r(c['oracle'] - chi * mask):10.2f} ({r(eo[0]):5.2f}/{r(eo[1]):5.2f}) | {r(c['dirty'] - c['oracle']):9.2f} ({r(dd[0]):5.2f}/{r(dd[1]):5.2f}) | {r(c['noise_clean'] - c['oracle']):7.2f} ({r(dn[0]):5.2f}/{r(dn[1]):5.2f})")
