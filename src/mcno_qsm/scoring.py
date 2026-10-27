"""Score a local-field map with the user's eNDI (ndiCG_auto), unchanged, from `external/`."""

import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np
import torch

from .physics import dipole_kernel
from .rc2 import phase_scale

EXTERNAL = Path(__file__).resolve().parents[2] / "external"


def _modules():
    if str(EXTERNAL) not in sys.path:
        sys.path.insert(0, str(EXTERNAL))
    import FANSI_tools
    import eNDI_module_torch

    return FANSI_tools, eNDI_module_torch


def limit_vram(device=None):
    """Hard per-process cap on the shared GPU from the env var MCNO_VRAM_GB (unset = no cap); exceeding it raises an OOM error."""
    gigabytes = os.environ.get("MCNO_VRAM_GB")
    if not gigabytes or not torch.cuda.is_available():
        return None
    index = device.index if getattr(device, "index", None) is not None else torch.cuda.current_device()
    total = torch.cuda.get_device_properties(index).total_memory
    fraction = min(1.0, float(gigabytes) * 2**30 / total)
    torch.cuda.set_per_process_memory_fraction(fraction, index)
    return fraction


def forward_field(chi, mask=None, voxel_size=(1.0, 1.0, 1.0)):
    """Unpadded forward model b = real(ifftn(D fftn(chi))) with the correctly aligned dipole kernel.

    FANSI_tools.dipole_kernel uses fftshift where ifftshift is needed, so for odd N its DC term sits at
    the wrong index (Sim2 is 164x205x205). For even N (the 256^3 padded eNDI grid) it is identical.
    """
    kernel = dipole_kernel(chi.shape, voxel_size, dtype=torch.float64).numpy()
    field = np.real(np.fft.ifftn(kernel * np.fft.fftn(chi))).astype(np.float32)
    return field if mask is None else field * mask


def masked_nrmse(estimate, reference, mask):
    """NRMSE in percent inside the mask (same convention as the eNDI module)."""
    return float(100 * np.linalg.norm(mask * (estimate - reference)) / np.linalg.norm(mask * reference))


def _pad_corner(volume, size):
    out = np.zeros((size,) * 3, dtype=np.float64)
    out[tuple(slice(0, n) for n in volume.shape)] = volume
    return out


def score_endi(field_ppm, chi_ppm, mask, weight, te=8e-3, b0=7.0, pad=256, alpha=1e-5,
               max_iter=150, tol_update=1e-10, is_precond=False, voxel_size=None, xsim_scale=1.0):
    """Run ndiCG_auto on `field_ppm` zero-padded (corner) to pad^3 and score against `chi_ppm` inside `mask`.

    Defaults are the notebook settings. `voxel_size` only feeds the automatic-stop masks and is left to the
    module default (1 mm) like the notebook; `xsim_scale` likewise (the notebook passes radians to XSIM).
    """
    tools, endi = _modules()
    limit_vram()
    shape = field_ppm.shape
    if max(shape) > pad:
        raise ValueError(f"volume {shape} does not fit in {pad}^3")
    scale = phase_scale(te, b0)
    kernel = tools.dipole_kernel((pad,) * 3, (pad,) * 3)  # 1 mm voxels, FOV = N
    params = {
        "K": kernel, "input": _pad_corner(field_ppm * scale, pad), "alpha": alpha,
        "isNonlinear": False, "weight": _pad_corner(weight, pad), "mask": _pad_corner(mask, pad),
        "maxOuterIter": max_iter, "tolUpdate": tol_update, "isPrecond": is_precond,
        "chi_cosmos": _pad_corner(chi_ppm * scale, pad), "isXSIM": True, "scale": xsim_scale,
    }
    if voxel_size is not None:
        params["voxelSize"] = np.array([voxel_size])
    with contextlib.redirect_stdout(io.StringIO()):
        out = endi.ndiCG_auto(params)
    crop = tuple(slice(0, n) for n in shape)
    chi_hat = out["x"][crop] / scale
    return {
        "chi": chi_hat.astype(np.float32),
        "iterations": int(out["iter"]),
        "seconds": float(out["time"]),
        "nrmse": masked_nrmse(chi_hat, chi_ppm, mask),
        "xsim": float(out["xsim_history"][-1]),
        "nrmse_history": [float(v) for v in out["rmse_history"]],
        "xsim_history": [float(v) for v in out["xsim_history"]],
    }
