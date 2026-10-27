"""S4d free check on Sim2 (no network): what does the dirty field look like OUTSIDE the brain mask?

psi = Frequency_ppm - D*Chi (whole-volume forward field, correct kernel). Reports rms of psi, of the true field and of
the magnitude weight per shell of distance (voxels) from the mask, and the fraction of the total field energy and of
the total psi energy that lies outside the mask.
Run from the project root: .venv\\Scripts\\python -I scripts/s4d_check_outside.py --data dataRC2
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt

from mcno_qsm.physics import dipole_kernel
from mcno_qsm.rc2 import load_rc2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataRC2")
    args = parser.parse_args()
    data = load_rc2(Path(args.data))
    chi, mask, voxel = data["chi"].astype(np.float64), data["mask"], data["voxel_size"]
    kernel = dipole_kernel(chi.shape, voxel, dtype=torch.float64).numpy()
    truth = np.real(np.fft.ifftn(kernel * np.fft.fftn(chi)))
    psi = data["field"] - truth
    distance = distance_transform_edt(~mask, sampling=voxel)
    weight = data["weight"]
    print(f"voxels in mask {mask.mean():.1%}; frequency rms in/out mask {data['field'][mask].std():.4f} / "
          f"{data['field'][~mask].std():.4f} ppm")
    print(f"energy share outside the mask: true field {np.sum(truth[~mask]**2) / np.sum(truth**2):.1%}, "
          f"psi {np.sum(psi[~mask]**2) / np.sum(psi**2):.1%}")
    print(f"{'shell (mm)':>12s} {'voxels':>9s} {'true rms':>9s} {'psi rms':>9s} {'psi/true':>8s} {'weight mean':>11s}")
    edges = [0, 1e-9, 3, 6, 10, 15, 25, 40, 1e9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        shell = (distance >= lo) & (distance < hi) if lo > 0 else mask
        if not shell.any():
            continue
        label = "in mask" if lo == 0 else f"{lo:g}-{min(hi, 999):g}"
        true_rms, psi_rms = np.sqrt((truth[shell] ** 2).mean()), np.sqrt((psi[shell] ** 2).mean())
        print(f"{label:>12s} {int(shell.sum()):9d} {true_rms:9.4f} {psi_rms:9.4f} {psi_rms / true_rms:8.2f} "
              f"{weight[shell].mean():11.3f}")
    # Is outside-psi noise-like? lag-1 correlation along z inside vs outside (white noise ~ 0).
    for name, region in (("in mask", mask), ("outside, 3-25 mm", (distance >= 3) & (distance < 25))):
        a, b = psi[..., :-1], psi[..., 1:]
        both = region[..., :-1] & region[..., 1:]
        print(f"psi lag-1 correlation along z, {name}: {np.corrcoef(a[both], b[both])[0, 1]:.3f}")


if __name__ == "__main__":
    main()
