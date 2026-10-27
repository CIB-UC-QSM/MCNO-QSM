"""Load the RC2 (Sim2) NIfTI volumes and derive the quantities the cleaner and the scorer share."""

from pathlib import Path

import nibabel as nib
import numpy as np

GAMMA_BAR = 42.58  # MHz/T
ECHO_TIMES = (4e-3, 12e-3, 20e-3, 28e-3)  # s, Magnitude.nii echoes


def hz_to_ppm(frequency, b0: float):
    return frequency / (GAMMA_BAR * b0)


def phase_scale(te: float, b0: float) -> float:
    """Radians per ppm of susceptibility field at echo time `te` (s) and field strength `b0` (T)."""
    return te * 2.0 * np.pi * GAMMA_BAR * b0


def magnitude_weight(magnitude, echo_times=ECHO_TIMES):
    """wPrime = sum_e(m_e^2 TE_e) / sum_e(m_e TE_e) with m = magnitude / global max; 0 where undefined."""
    m = magnitude / magnitude.max()
    te = np.asarray(echo_times, dtype=m.dtype)
    numerator = np.sum(m**2 * te, axis=-1)
    denominator = np.sum(m * te, axis=-1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def load_rc2(root, b0: float = 7.0) -> dict:
    """Return float32 arrays; `field` is the dirty Frequency.nii converted to ppm (not masked)."""
    root = Path(root)
    chi_image = nib.load(root / "Chi.nii")
    chi = chi_image.get_fdata(dtype=np.float32)
    frequency = nib.load(root / "Frequency.nii").get_fdata(dtype=np.float32)
    magnitude = nib.load(root / "Magnitude.nii").get_fdata(dtype=np.float32)
    mask = nib.load(root / "MaskBrainExtracted.nii").get_fdata(dtype=np.float32) > 0
    if not (chi.shape == frequency.shape == mask.shape == magnitude.shape[:3]):
        raise ValueError("RC2 volumes must share one grid")
    if not all(np.isfinite(a).all() for a in (chi, frequency, magnitude)):
        raise ValueError("RC2 volumes must be finite")
    return {
        "chi": chi,
        "field": hz_to_ppm(frequency, b0).astype(np.float32),
        "magnitude": magnitude,
        "weight": magnitude_weight(magnitude).astype(np.float32),
        "mask": mask,
        "voxel_size": tuple(float(v) for v in chi_image.header.get_zooms()[:3]),
        "affine": chi_image.affine,
    }
