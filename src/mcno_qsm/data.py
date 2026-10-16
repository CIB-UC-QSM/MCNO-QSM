"""Load the original Sim2 arrays without resampling or fitting their alignment."""

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.io import loadmat
from torch import Tensor


@dataclass
class Sim2:
    field: Tensor
    ground_truth: Tensor
    mask: Tensor
    voxel_size: tuple[float, float, float]
    reference: nib.Nifti1Image
    input_scale: float


def load_sim2(field, ground_truth, mask, crop_start=(0, 0, 0), field_scale=1.0, **_) -> Sim2:
    reference = nib.load(ground_truth)
    chi = reference.get_fdata(dtype=np.float32)
    support = np.asarray(loadmat(mask)["mask_final"])
    measured = np.asarray(loadmat(field)["phase"])
    if chi.ndim != 3 or support.shape != chi.shape or measured.ndim != 3:
        raise ValueError("expected 3D field and matching 3D ground truth and mask")
    if len(crop_start) != 3 or any(not isinstance(v, int) or v < 0 for v in crop_start):
        raise ValueError("crop_start must contain three nonnegative integers")
    if any(start + size > n for start, size, n in zip(crop_start, chi.shape, measured.shape)):
        raise ValueError("field does not cover the requested crop")
    crop = tuple(slice(start, start + n) for start, n in zip(crop_start, chi.shape))
    measured = np.asarray(measured[crop], dtype=np.float32) * field_scale
    if not all(np.isfinite(x).all() for x in (measured, chi, support)):
        raise ValueError("input arrays must be finite")
    support = support > 0
    if not support.any() or np.linalg.norm(chi[support]) == 0:
        raise ValueError("mask must be nonempty and GT must have nonzero masked norm")
    scale = float(measured[support].std())
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("masked input field must have nonzero standard deviation")

    def tensor(x):
        return torch.from_numpy(np.array(x, copy=True))[None, None]

    return Sim2(tensor(measured), tensor(chi), tensor(support),
                tuple(float(v) for v in reference.header.get_zooms()[:3]), reference, scale)


def save_volume(path: Path, volume: Tensor, reference: nib.Nifti1Image) -> None:
    data = volume.detach().cpu().numpy().reshape(reference.shape).astype(np.float32)
    image = nib.Nifti1Image(data, reference.affine, reference.header.copy())
    image.set_data_dtype(np.float32)
    nib.save(image, path)
