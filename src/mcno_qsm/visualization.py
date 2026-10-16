"""Central orthogonal slices and per-update training curves."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .metrics import masked_nrmse


def plot_slices(volume, path: str | Path, *, title="Susceptibility", limits=(-0.1, 0.1),
                ground_truth=None, mask=None, voxel_size=(1.0, 1.0, 1.0), difference=False,
                cmap="gray"):
    """Save central sagittal/coronal/axial slices; NRMSE uses the full mask.

    Inputs must be 3D. Differences are signed prediction minus ground truth.
    Provide `difference=True` for the default difference contrast [-.03,.03].
    """
    def array(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    data = array(volume)
    if data.ndim != 3:
        raise ValueError("plot_slices expects a 3D volume")
    if ground_truth is not None and mask is not None:
        score = masked_nrmse(torch.as_tensor(data), torch.as_tensor(array(ground_truth)),
                             torch.as_tensor(array(mask), dtype=torch.bool)).item()
        title += f" | NRMSE = {score:.6f}"
    if difference:
        limits = (-0.03, 0.03)
    center = tuple(n // 2 for n in data.shape)
    slices = (data[center[0], :, :], data[:, center[1], :], data[:, :, center[2]])
    aspects = (voxel_size[2] / voxel_size[1], voxel_size[2] / voxel_size[0],
               voxel_size[1] / voxel_size[0])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    for ax, image, name, aspect in zip(axes, slices, ("Sagittal", "Coronal", "Axial"), aspects):
        im = ax.imshow(image.T, origin="lower", cmap=cmap, aspect=aspect,
                       vmin=limits[0], vmax=limits[1])
        ax.set_title(name)
        ax.axis("off")
    fig.suptitle(title)
    fig.colorbar(im, ax=axes, label="ppm", shrink=0.8)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_history(rows: list[dict], path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    ax.plot([r["step"] for r in rows], [r["nrmse"] for r in rows])
    ax.set(xlabel="Optimizer updates", ylabel="Masked NRMSE", title="Sim2 overfit", ylim=(0, 1))
    ax.grid(alpha=0.3)
    fig.savefig(path, dpi=160)
    plt.close(fig)
