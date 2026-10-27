"""AHEAD (7 T, processed QSM maps) -> B0-aligned 1 mm isotropic volumes, one interpolation per volume.

The NIfTI affines of AHEAD are scanner-anatomical (sform/qform code 1) and the tilt differs between subjects (CLAUDE.md 9e-update),
so world z = B0. Resampling every volume with its own affine into a grid whose axes ARE the world axes puts B0 along z for the repo
dipole kernel and removes flips / tilts in a single interpolation. Only numpy / scipy / nibabel are needed (no torch).
"""

from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

TARGET_MM = 1.0
RC1_WEIGHT_MEAN = 0.264  # in-mask mean of RC1 `magn`; Sim2 wPrime 0.252. The nets are sensitive to the scale of the weight channel (9u/9v).
MODALITIES = {"qsm": "qsm", "r2": "r2starmap", "t1w": "t1w"}


def find_subjects(root) -> dict[str, dict[str, Path]]:
    """{subject id: {'qsm': path, 'r2': path, 't1w': path}} for every subject that has all three modalities."""
    root = Path(root)
    found: dict[str, dict[str, Path]] = {}
    for key, folder in MODALITIES.items():
        for path in sorted((root / folder).glob(f"*mod-{folder}_*.nii.gz")):
            subject = path.name.split("_")[0]
            found.setdefault(subject, {})[key] = path
    return {s: p for s, p in sorted(found.items()) if set(p) == set(MODALITIES)}


def world_grid(affine, shape, support, margin_mm=4.0, spacing=TARGET_MM, minimum=96):
    """Origin (world mm) and output shape of an axis-aligned `spacing` grid covering the voxels where `support` is True."""
    idx = np.argwhere(support)
    lo, hi = idx.min(0), idx.max(0)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])], dtype=float)
    world = corners @ affine[:3, :3].T + affine[:3, 3]
    low, high = world.min(0) - margin_mm, world.max(0) + margin_mm
    out_shape = np.maximum(np.ceil((high - low) / spacing).astype(int) + 1, minimum)
    origin = (low + high) / 2 - (out_shape - 1) * spacing / 2
    return origin, tuple(int(n) for n in out_shape)


def resample_to_world(volume, affine, origin, out_shape, spacing=TARGET_MM, order=1, antialias=True):
    """Resample `volume` (voxel -> world by `affine`) onto the grid origin + spacing * index (axes = world axes).

    A Gaussian anti-alias filter with sigma = 0.5 sqrt(f^2 - 1) input voxels (f = spacing / voxel size > 1) is applied first.
    """
    volume = np.asarray(volume, dtype=np.float32)
    rotation, translation = affine[:3, :3], affine[:3, 3]
    zooms = np.linalg.norm(rotation, axis=0)
    if antialias:
        factor = spacing / zooms
        sigma = 0.5 * np.sqrt(np.maximum(factor**2 - 1.0, 0.0))
        volume = ndimage.gaussian_filter(volume, sigma, mode="nearest")
    matrix = np.linalg.inv(rotation)
    offset = matrix @ (np.asarray(origin) - translation)
    return ndimage.affine_transform(volume, matrix * spacing, offset, output_shape=out_shape, order=order, mode="constant", cval=0.0)


def b0_in_voxel_axes(affine):
    """World z (B0) expressed in the voxel axes of the source volume (unit vector) and its tilt from the nearest axis in degrees."""
    zooms = np.linalg.norm(affine[:3, :3], axis=0)
    b0 = np.linalg.inv(affine[:3, :3] / zooms) @ np.array([0.0, 0.0, 1.0])
    b0 = b0 / np.linalg.norm(b0)
    return b0, float(np.degrees(np.arccos(np.clip(np.abs(b0).max(), -1, 1))))


def quantile_match(values, reference_quantiles):
    """Map `values` (1-D) to the distribution given by `reference_quantiles` (sorted, evenly spaced probabilities 0..1), keeping ranks."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size)
    ranks[order] = np.linspace(0.0, 1.0, values.size)
    return np.interp(ranks, np.linspace(0.0, 1.0, len(reference_quantiles)), reference_quantiles).astype(np.float32)


def preprocess_subject(paths, te_ms=20.0, erode=1, weight_mean=RC1_WEIGHT_MEAN, weight_reference=None):
    """Return the 1 mm B0-aligned arrays and a summary dict for one subject.

    The weight proxy is T1w * exp(-TE R2*). By default it is scaled to the in-mask mean `weight_mean`; with `weight_reference`
    (quantiles of RC1's in-mask `magn`) it is rank-matched to that distribution instead, because the raw proxy is much wider
    (p1 0.002-0.02, p99 0.45-0.55 against 0.13-0.38) and the training noise is proportional to 1 / weight.
    """
    qsm_image = nib.load(paths["qsm"])
    affine = qsm_image.affine
    qsm = qsm_image.get_fdata(dtype=np.float32)
    r2 = nib.load(paths["r2"]).get_fdata(dtype=np.float32)
    t1w = nib.load(paths["t1w"]).get_fdata(dtype=np.float32)
    if not (qsm.shape == r2.shape == t1w.shape):
        raise ValueError(f"{paths['qsm'].name}: qsm / r2star / t1w must share one grid")
    for name, array in (("qsm", qsm), ("r2", r2), ("t1w", t1w)):
        if not np.isfinite(array).all():
            raise ValueError(f"{paths['qsm'].name}: non-finite values in {name}")
    for name in ("r2", "t1w"):
        other = nib.load(paths[name]).affine
        if not np.allclose(other, affine, atol=1e-3):
            raise ValueError(f"{paths['qsm'].name}: affine of {name} differs from qsm")
    inside = qsm != 0
    origin, out_shape = world_grid(affine, qsm.shape, inside)

    mask_soft = resample_to_world(inside.astype(np.float32), affine, origin, out_shape)
    chi_sum = resample_to_world(qsm * inside, affine, origin, out_shape)
    mask = mask_soft > 0.5
    if erode:
        mask = ndimage.binary_erosion(mask, iterations=erode)
    chi = np.where(mask, chi_sum / np.maximum(mask_soft, 0.5), 0.0).astype(np.float32)   # normalized interpolation: no dilution at the edge
    chi[mask] -= chi[mask].mean()

    r2_w = resample_to_world(np.clip(r2, 0.0, None), affine, origin, out_shape)
    t1_w = resample_to_world(t1w, affine, origin, out_shape)
    proxy = np.clip(t1_w, 0.0, None) * np.exp(-te_ms * r2_w)         # R2* in 1/ms
    if weight_reference is None:
        weight = np.clip(proxy * (weight_mean / proxy[mask].mean()), 0.0, 1.0).astype(np.float32)
    else:
        weight = np.zeros_like(proxy, dtype=np.float32)
        weight[mask] = quantile_match(proxy[mask], weight_reference)

    b0, tilt = b0_in_voxel_axes(affine)
    p = lambda a, q: float(np.percentile(a[mask], q))
    summary = {
        "source_shape": list(qsm.shape), "source_zooms": np.linalg.norm(affine[:3, :3], axis=0).round(4).tolist(),
        "out_shape": list(out_shape), "origin_world_mm": np.asarray(origin).round(3).tolist(),
        "b0_in_source_voxel_axes": b0.round(4).tolist(), "tilt_deg": round(tilt, 2),
        "mask_voxels": int(mask.sum()), "mask_fraction": float(mask.mean()),
        "chi_p0.1": p(chi, 0.1), "chi_p99.9": p(chi, 99.9), "chi_std": float(chi[mask].std()),
        "weight_mean": float(weight[mask].mean()), "weight_p1": p(weight, 1), "weight_p99": p(weight, 99),
    }
    return {"chi": chi, "mask": mask, "weight": weight}, summary


def save_subject(path, arrays, summary):
    np.savez_compressed(path, chi=arrays["chi"].astype(np.float16), weight=arrays["weight"].astype(np.float16),
                        mask=arrays["mask"].astype(np.uint8), voxel_size=np.array([TARGET_MM] * 3, dtype=np.float32),
                        summary=np.array(str(summary)))


def load_subject(path):
    """float32 chi / weight, bool mask, voxel size (1 mm) from a cached npz."""
    z = np.load(path)
    return z["chi"].astype(np.float32), z["weight"].astype(np.float32), z["mask"] > 0, tuple(float(v) for v in z["voxel_size"])
