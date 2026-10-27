"""Figures of the technical report computed from the project code (run from MCNO-QSM:
    uv run python -I technicalReport/figures/make_figures.py)."""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from mcno_qsm.dipolelets import SIM2_DIPOLELET, create_dipolelet_windows

OUT = Path(__file__).parent
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150,
                     "axes.spines.top": False, "axes.spines.right": False})
SIGMA, J, DELTA = 0.5, 3, 0.05
THRESHOLDS = np.logspace(np.log10(DELTA), np.log10(1 / 3), 2).tolist()


def angular_windows(magnitude):
    return create_dipolelet_windows(torch.as_tensor(magnitude, dtype=torch.float64), THRESHOLDS, None,
                                    window="gaussian", width_factor=0.5).numpy()


def radial(kappa):
    """R_j(k) and R_res(k) of the paper for sigma_j = sigma 2^j; kappa = |k| in rad/voxel."""
    g = [np.exp(-0.5 * (SIGMA * 2**j) ** 2 * kappa**2) for j in range(J)]
    r = [np.prod(g[:j], axis=0) * (1 - g[j]) if j else 1 - g[0] for j in range(J)]
    return r, np.prod(g, axis=0)


def sampled_gaussian_response(sigma, kappa):
    """Frequency response (one axis) of the implemented filter: sampled Gaussian truncated at 4 sigma, normalised."""
    radius = int(4.0 * sigma + 0.5)
    taps = np.arange(-radius, radius + 1)
    w = np.exp(-0.5 * (taps / sigma) ** 2)
    w /= w.sum()
    return (w[None, :] * np.cos(np.outer(kappa, taps))).sum(1)


def radial_sampled(kappa):
    g = [sampled_gaussian_response(SIGMA * 2**j, kappa) for j in range(J)]
    return [np.prod(g[:j], axis=0) * (1 - g[j]) if j else 1 - g[0] for j in range(J)], np.prod(g, axis=0)


def filters():
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 2.9))
    kappa = np.linspace(0, math.pi, 400)
    r, res = radial(kappa)
    for j, rj in enumerate(r):
        axes[0].plot(kappa, rj, label=rf"$R_{j}$ ($\sigma_{j}={SIGMA * 2**j:g}$)")
    axes[0].plot(kappa, res, "k--", label=r"$R_{\mathrm{res}}$")
    axes[0].plot(kappa, sum(r) + res, color="gray", lw=0.8, label=r"$\sum_j R_j+R_{\mathrm{res}}$")
    rs, ress = radial_sampled(kappa)
    for j, rj in enumerate(rs):
        axes[0].plot(kappa, rj, color=f"C{j}", ls=(0, (2, 2)), lw=1.1)
    axes[0].plot([], [], color="gray", ls=(0, (2, 2)), label="as implemented (sampled kernel)")
    axes[0].set(xlabel=r"$\|\mathbf{k}\|$ (rad/voxel)", ylabel="filter response", title="radial filters")
    axes[0].legend(fontsize=6.5)
    d = np.linspace(0, 2 / 3, 600)
    a = angular_windows(d)
    for m in range(3):
        axes[1].plot(d, a[m], label=rf"$A_{m}$")
    axes[1].plot(d, a.sum(0), color="gray", lw=0.8, label=r"$\sum_m A_m$")
    for t in THRESHOLDS:
        axes[1].axvline(t, color="k", ls=":", lw=0.8)
    axes[1].set(xlabel=r"$|D(\mathbf{k})|$", title=r"angular windows ($\delta_0=0.05,\ \delta_1=1/3$)")
    axes[1].legend(fontsize=7)
    theta = np.linspace(0, 90, 600)
    dd = np.abs(1 / 3 - np.cos(np.radians(theta)) ** 2)
    a = angular_windows(dd)
    for m in range(3):
        axes[2].plot(theta, a[m], label=rf"$A_{m}$")
    axes[2].axvline(math.degrees(math.acos(1 / math.sqrt(3))), color="r", ls="--", lw=1, label=r"magic angle $54.7^\circ$")
    axes[2].set(xlabel=r"polar angle $\theta$ of $\mathbf{k}$ w.r.t. $B_0$ (deg)", title="the same windows against the angle")
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_filters.pdf")
    plt.close(fig)


def kspace():
    n = 129
    freqs = [np.fft.fftshift(np.fft.fftfreq(n)) * 2 * math.pi] * 3
    kx, ky, kz = np.meshgrid(*freqs, indexing="ij")
    k2 = kx**2 + ky**2 + kz**2
    d = 1 / 3 - kz**2 / np.maximum(k2, 1e-12)
    kappa = np.sqrt(k2)
    r, res = radial(kappa)
    a = angular_windows(np.abs(d))
    mid = n // 2
    fig, axes = plt.subplots(3, 4, figsize=(9.4, 7.0))
    for j in range(J):
        for m in range(3):
            axes[j, m].imshow((r[j] * a[m])[:, mid, :].T, origin="lower", cmap="magma", vmin=0, vmax=1,
                              extent=(-math.pi, math.pi, -math.pi, math.pi))
            axes[j, m].set_title(rf"$R_{j}A_{m}$", fontsize=9)
        axes[j, 3].axis("off")
    axes[0, 3].imshow(res[:, mid, :].T, origin="lower", cmap="magma", vmin=0, vmax=1,
                      extent=(-math.pi, math.pi, -math.pi, math.pi))
    axes[0, 3].axis("on")
    axes[0, 3].set_title(r"$R_{\mathrm{res}}$ (not split)", fontsize=9)
    total = sum(r[j] * a[m] for j in range(J) for m in range(3)) + res
    axes[1, 3].imshow(total[:, mid, :].T, origin="lower", cmap="gray", vmin=0, vmax=1.2,
                      extent=(-math.pi, math.pi, -math.pi, math.pi))
    axes[1, 3].axis("on")
    axes[1, 3].set_title(rf"sum of the 10 windows (max err {abs(total - 1).max():.1e})", fontsize=7.5)
    for ax in axes.ravel():
        ax.set_xlabel(r"$k_x$", fontsize=7)
        ax.set_ylabel(r"$k_z$", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig_kspace_segments.pdf")
    plt.close(fig)
    return float(abs(total - 1).max())


def memory():
    sizes = {"96": (2.68, 5.10, 4.28, 8.58), "112": (4.24, 8.08, 6.79, 13.61), "128": (6.29, 12.0, 10.10, 20.24)}
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    x = np.arange(3)
    labels = [f"{s}$^3$" for s in sizes]
    for i, (name, col) in enumerate((("per-segment + per-block checkpoint, batch 1", 0), ("same, batch 2", 1),
                                     ("per-block checkpoint only, batch 1", 2), ("same, batch 2", 3))):
        ax.bar(x + (i - 1.5) * 0.2, [v[col] for v in sizes.values()], 0.2, label=name)
    ax.axhline(32, color="r", ls="--", lw=1)
    ax.text(0.02, 29.5, "GPU 32 GB (shared)", color="r", fontsize=7)
    ax.set(xticks=x, xticklabels=labels, ylabel="peak GPU memory (GB)", xlabel="training patch", ylim=(0, 34))
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(0, 0.93))
    fig.tight_layout()
    fig.savefig(OUT / "fig_memory.pdf")
    plt.close(fig)


if __name__ == "__main__":
    filters()
    print("partition-of-unity max error in k space:", kspace())
    memory()
