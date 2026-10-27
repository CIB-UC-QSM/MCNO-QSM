"""G2 (CLAUDE.md 9ap): linear cone-consistency inpainting of the calcification shell (<=3 mm of the 37 voxels, 495 voxels).
Unknown x on the shell S; minimise ||C (f0 + P_S x)||^2 + mu ||x||^2, C = Fourier multiplier w(|D|) (w = 1 in the cone), f0 = a field with the
shell zeroed. Normal equations 495x495 built from the kernel kappa = ifft(w^2) (no per-column FFT). Reported: relative error of the shell
field ||x - b_S|| / ||b_S|| and gain <x, b_S>/||b_S||^2 (criterion of 9am: error < 0.8, gain > 0.5; noise_clean has 0.986 / 0.291).
Surroundings: the TRUE field (upper bound) or the noise_clean field (realistic)."""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcno_qsm.physics import dipole_kernel  # noqa: E402
from mcno_qsm.rc2 import load_rc2  # noqa: E402
from mcno_qsm.scoring import forward_field  # noqa: E402

data = load_rc2("dataRC2", 7.0)
mask, chi, voxel = data["mask"].astype(np.float32), data["chi"], data["voxel_size"]
shape = chi.shape
b = forward_field(chi, mask, voxel)
clean = nib.load("runs/s4b_noise_clean/best_clean.nii.gz").get_fdata(dtype=np.float32) * mask
dirty = data["field"] * mask
calc = chi < -0.5
dist = ndimage.distance_transform_edt(~calc, sampling=voxel)
S = (dist <= 3.0) & (mask > 0)
idx = np.argwhere(S)
print("shell voxels", len(idx), "rms b in shell", float(np.sqrt((b[S] ** 2).mean())))
adk = np.abs(dipole_kernel(shape, voxel, dtype=torch.float64).numpy())
sh = np.array(shape)
for d0 in (0.03, 0.05, 0.1, 0.2):
    w = np.exp(-(adk / d0) ** 4).astype(np.float64)       # ~1 inside |D|<d0, smooth edge
    kappa = np.real(np.fft.ifftn(w ** 2))                 # C^T C kernel (circular)
    diff = (idx[:, None, :] - idx[None, :, :]) % sh
    M = kappa[diff[..., 0], diff[..., 1], diff[..., 2]]
    for ctx_name, ctx in (("true surroundings", b), ("noise_clean surroundings", clean), ("dirty surroundings", dirty)):
        f0 = ctx.copy().astype(np.float64); f0[S] = 0
        rhs = -np.real(np.fft.ifftn(w ** 2 * np.fft.fftn(f0)))[S]
        out = []
        for mu in (1e-6, 1e-4, 1e-2):
            x = np.linalg.solve(M + mu * np.eye(len(idx)) * np.trace(M) / len(idx), rhs)
            bs = b[S]
            out.append(f"mu {mu:.0e}: err {np.linalg.norm(x - bs) / np.linalg.norm(bs):.3f} gain {np.dot(x, bs) / np.dot(bs, bs):.3f}")
        print(f"d0={d0:<5} {ctx_name:26s} cond(M)={np.linalg.cond(M):.1e} | " + " | ".join(out), flush=True)
