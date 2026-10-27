"""Intravoxel dephasing of a strong compact source: fine-grid field, complex block average, coarse phase and magnitude.

A coarse voxel signal is the mean of the fine-grid signals inside it, so where the field varies strongly within one voxel the
magnitude drops and the phase of the mean departs from the field of the block-averaged susceptibility (the training target).
"""

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .physics import dipole_kernel


def fine_field(chi_fine: Tensor, voxel_fine, pad: int = 32) -> Tensor:
    """D * chi on the fine grid (B0 along z), zero-padded so that the periodic wrap stays far from the patch. `voxel_fine`: mm or a triple."""
    voxel = (voxel_fine,) * 3 if np.isscalar(voxel_fine) else tuple(voxel_fine)
    padded = F.pad(chi_fine, (pad,) * 6)
    kernel = dipole_kernel(tuple(padded.shape), voxel, device=chi_fine.device, dtype=chi_fine.dtype)
    field = torch.fft.ifftn(torch.fft.fftn(padded) * kernel).real
    return field[pad:-pad, pad:-pad, pad:-pad]


def block_mean(x: Tensor, factor: int) -> Tensor:
    return F.avg_pool3d(x[None, None], factor, factor)[0, 0]


def source_chi_fine(shape_coarse, factor: int, centre, radii, amplitude: float, angles_deg=(0.0, 0.0, 0.0)) -> Tensor:
    """Ellipsoid of susceptibility `amplitude` (ppm) on the fine grid; `centre` and `radii` in COARSE voxels (fractions allowed)."""
    n = [s * factor for s in shape_coarse]
    axes = [(torch.arange(k, dtype=torch.float64) + 0.5) / factor for k in n]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"))
    centre = torch.as_tensor(centre, dtype=torch.float64).reshape(3, 1, 1, 1) + 0.5
    from .artifacts import rotation_matrix
    rotation = torch.as_tensor(rotation_matrix(angles_deg).T, dtype=torch.float64)
    local = torch.einsum("ij,jxyz->ixyz", rotation, grid - centre)
    radii = torch.as_tensor(radii, dtype=torch.float64).reshape(3, 1, 1, 1)
    inside = ((local / radii).square().sum(0) <= 1.0).double()
    return amplitude * inside


def dephase(background_ppm: Tensor, magnitude: Tensor, chi_fine_source: Tensor, factor: int, voxel_mm,
            te: float, phs_per_ppm_per_s: float, r2star: float = 0.0, intrinsic_dip: Tensor | None = None):
    """Complex block average of the fine-grid signal for one echo time.

    background_ppm / magnitude : coarse (D*chi of the head, magnitude at this TE before the source), upsampled trilinearly;
    chi_fine_source            : fine-grid susceptibility of the added source (ppm);
    returns (coarse_signal complex, fine-block-averaged source+background ppm field (the 'target' field) ).
    """
    voxel = (voxel_mm,) * 3 if np.isscalar(voxel_mm) else tuple(voxel_mm)
    fine_src = fine_field(chi_fine_source, tuple(v / factor for v in voxel))
    up = lambda x: F.interpolate(x[None, None], scale_factor=factor, mode="trilinear", align_corners=False)[0, 0]
    background_fine = up(background_ppm)
    magnitude_fine = up(magnitude)
    if intrinsic_dip is not None:
        magnitude_fine = magnitude_fine * intrinsic_dip
    total = background_fine + fine_src
    signal = magnitude_fine * torch.exp(1j * total * (te * phs_per_ppm_per_s))
    if r2star:
        signal = signal * np.exp(-te * r2star)
    return block_mean(signal.real, factor) + 1j * block_mean(signal.imag, factor), block_mean(total, factor)


def box_echoes(background_ppm: Tensor, magnitudes: Tensor, chi_fine_source: Tensor, factor: int, voxel, tes, per_ppm_s: float):
    """Coarse complex echoes [E, B, B, B] of a box holding one source, and the block-averaged susceptibility of the source.

    background_ppm [B,B,B] is the coarse field of everything else, magnitudes [E,B,B,B] the echo magnitudes before dephasing; both are upsampled
    trilinearly. The fine source field is computed once and reused for every echo.
    """
    voxel = (voxel,) * 3 if np.isscalar(voxel) else tuple(voxel)
    up = lambda x: F.interpolate(x[None, None], scale_factor=factor, mode="trilinear", align_corners=False)[0, 0]
    total = up(background_ppm) + fine_field(chi_fine_source, tuple(v / factor for v in voxel))
    echoes = []
    for te, magnitude in zip(tes, magnitudes):
        signal = up(magnitude) * torch.exp(1j * total * (te * per_ppm_s))
        echoes.append(torch.complex(block_mean(signal.real, factor), block_mean(signal.imag, factor)))
    return torch.stack(echoes), block_mean(chi_fine_source, factor)
