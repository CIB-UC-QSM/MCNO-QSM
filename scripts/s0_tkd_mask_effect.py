"""S0: how much does the brain mask alone limit an exact inversion of the true clean field? (TKD, no network)

Inverts D*chi_gt with TKD for decreasing thresholds, (a) over the whole volume and (b) after multiplying by the mask
(the training target), with the correctly aligned kernel and with the FANSI kernel (misaligned for odd N).
Run from the project root: python -I scripts/s0_tkd_mask_effect.py --data dataRC2
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "external"))
from FANSI_tools import dipole_kernel as fansi_kernel  # noqa: E402

from mcno_qsm.physics import dipole_kernel  # noqa: E402
from mcno_qsm.rc2 import load_rc2  # noqa: E402


def tkd(field, kernel, threshold):
    inverse = np.where(np.abs(kernel) >= threshold, 1 / np.where(kernel == 0, 1, kernel), np.sign(kernel) / threshold)
    return np.real(np.fft.ifftn(np.fft.fftn(field) * inverse))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataRC2")
    args = parser.parse_args()
    data = load_rc2(args.data)
    chi, mask, voxel = data["chi"].astype(np.float64), data["mask"], data["voxel_size"]
    shape = chi.shape
    good = dipole_kernel(shape, voxel, dtype=torch.float64).numpy()
    fansi = np.asarray(fansi_kernel(shape, np.array(shape) * np.array(voxel)))
    nrmse = lambda est: 100 * np.linalg.norm(mask * (est - chi)) / np.linalg.norm(mask * chi)
    full = np.real(np.fft.ifftn(good * np.fft.fftn(chi)))
    print("odd axes:", [n % 2 for n in shape], "| chi rms inside / outside mask: %.3f / %.3f ppm" % (
        np.sqrt((chi[mask] ** 2).mean()), np.sqrt((chi[~mask] ** 2).mean())))
    print(f"{'threshold':>9s} | {'whole field':>11s} | {'mask*field':>10s} | {'mask*field, FANSI kernel':>24s}")
    for threshold in [0.1, 0.03, 0.01, 1e-3, 1e-4, 1e-6]:
        print(f"{threshold:9g} | {nrmse(tkd(full, good, threshold)):10.2f}% | {nrmse(tkd(full * mask, good, threshold)):9.2f}% | "
              f"{nrmse(tkd(full * mask, fansi, threshold)):23.2f}%")


if __name__ == "__main__":
    main()
