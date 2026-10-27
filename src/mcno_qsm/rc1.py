"""Load RC1 (Challenge 1, FANSI data): one COSMOS volume with a real in-vivo tissue phase and STI tensors."""

from pathlib import Path

import numpy as np
import scipy.io as sio
import torch

from .rc2 import phase_scale
from .scoring import forward_field

TE = 25e-3  # s
B0 = 2.8936  # T


def _mat(root: Path, name: str, key: str | None = None) -> np.ndarray:
    return sio.loadmat(root / f"{name}.mat")[key or name]


def load_rc1(root) -> dict:
    """float32 arrays in ppm. `chi` has its global mean removed (as the FANSI code does)."""
    root = Path(root)
    chi = _mat(root, "chi_cosmos").astype(np.float32)
    chi = chi - chi.mean()
    sti = sio.loadmat(root / "chi_sti.mat")
    mask = _mat(root, "msk") > 0
    magnitude = _mat(root, "magn").astype(np.float32)
    tissue = _mat(root, "phs_tissue").astype(np.float32)
    voxel_size = tuple(float(v) for v in _mat(root, "spatial_res")[0])
    if not (chi.shape == mask.shape == magnitude.shape == tissue.shape):
        raise ValueError("RC1 volumes must share one grid")
    if not all(np.isfinite(a).all() for a in (chi, magnitude, tissue)):
        raise ValueError("RC1 volumes must be finite")
    return {
        "chi": chi, "mask": mask, "magnitude": magnitude, "phs_tissue": tissue,
        "chi_13": sti["chi_13"].astype(np.float32), "chi_23": sti["chi_23"].astype(np.float32),
        "chi_33": sti["chi_33"].astype(np.float32),
        "voxel_size": voxel_size, "te": TE, "b0": B0, "phs_scale": phase_scale(TE, B0),
    }


def sti_fields(chi_13, chi_23, chi_33, voxel_size):
    """Fields of the STI components (notebook cell 5): p33 with the dipole kernel, p13 and p23 with kx kz/k^2, ky kz/k^2.

    Returns (pSTI, pQ) with pQ = p13 + p23 and pSTI = p33 - pQ. Physical frequencies replace the notebook's
    normalized grid (same ratios); DC is set to zero.
    """
    shape = chi_33.shape
    freqs = [torch.fft.fftfreq(n, d=v, dtype=torch.float64) for n, v in zip(shape, voxel_size)]
    kx, ky, kz = torch.meshgrid(*freqs, indexing="ij")
    k2 = (kx.square() + ky.square() + kz.square()).clamp_min(torch.finfo(torch.float64).eps)

    def apply(kernel, volume):
        spectrum = torch.fft.fftn(torch.from_numpy(volume).double())
        return torch.fft.ifftn(kernel * spectrum).real.numpy().astype(np.float32)

    p33 = forward_field(chi_33, None, voxel_size)
    p13 = apply(kx * kz / k2, chi_13)
    p23 = apply(ky * kz / k2, chi_23)
    pq = p13 + p23
    return p33 - pq, pq


def clean_field(chi, voxel_size, mask=None):
    """The training target before masking: D * chi with the repo kernel (B0 along z)."""
    return forward_field(chi, mask, voxel_size)

