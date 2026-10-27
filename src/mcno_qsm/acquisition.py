"""Multi-echo acquisition model and noise statistics for the training families `acq` and `intravoxel` (CLAUDE.md 9bp).

Echo signals are `mag_e * exp(i * 2 pi gamma B0 TE_e * b)` plus complex noise; the dirty field is the weighted least-squares slope of the phase
unwrapped along the echoes (the way Sim2's Frequency.nii behaves near its calcification: the estimate aliases beyond pi / (2 pi gamma B0 dTE),
about 0.21 ppm for 4/12/20/28 ms at 7 T). Noise statistics beyond the notebook's iid complex Gaussian: spatially correlated, heavy-tailed
(Student-t) and spatially varying (smooth gain map) noise, all with unit power per component before the 1/SNR scaling.
"""

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

GAMMA_BAR = 42.58  # MHz/T
NOISE_KINDS = ("gaussian", "colored", "heavy", "hetero")


def draw_noise_spec(rng, cfg):
    """Noise statistics of one sample, or None (iid Gaussian, no rng draw) when `cfg.noise_kinds` is not set."""
    if not cfg.noise_kinds:
        return None
    kinds = [k for k, _ in cfg.noise_kinds]
    probabilities = np.array([p for _, p in cfg.noise_kinds], dtype=float)
    kind = str(rng.choice(kinds, p=probabilities / probabilities.sum()))
    if kind not in NOISE_KINDS:
        raise ValueError(f"unknown noise kind {kind!r}; choose from {NOISE_KINDS}")
    spec = {"kind": kind}
    if kind == "colored":
        spec["sigma"] = float(rng.uniform(*cfg.noise_colored))
    elif kind == "heavy":
        spec["df"] = int(rng.integers(cfg.noise_heavy_df[0], cfg.noise_heavy_df[1] + 1))
    elif kind == "hetero":
        spec["depth"] = float(rng.uniform(*cfg.noise_hetero_depth))
    return spec


def _gaussian_blur(x: Tensor, sigma: float) -> tuple[Tensor, float]:
    """Separable Gaussian blur of [C, X, Y, Z]; also returns the power gain sum(k^2)^3 so the caller can restore unit variance."""
    radius = max(1, int(np.ceil(3 * sigma)))
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel = kernel / kernel.sum()
    for axis in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = -1
        padding = [0, 0, 0]
        padding[axis] = radius
        x = F.conv3d(x[:, None], kernel.reshape(shape), padding=padding)[:, 0]
    return x, float(kernel.square().sum() ** 3)


def complex_noise(shape, spec, generator, device, dtype=torch.float32) -> Tensor:
    """Complex noise of shape `shape` = (..., X, Y, Z) with unit variance per component (real and imaginary parts independent).

    `spec` comes from draw_noise_spec (None = iid Gaussian). Leading dimensions (echoes) are independent except for the hetero gain map,
    which is shared like a coil sensitivity.
    """
    kind = "gaussian" if spec is None else spec["kind"]
    draw = lambda *s: torch.randn(s, generator=generator, device=device, dtype=dtype)
    lead, volume = tuple(shape[:-3]), tuple(shape[-3:])
    if kind == "gaussian":
        n = draw(2, *shape)
    elif kind == "colored":
        n, gain = _gaussian_blur(draw(2, *shape).reshape(-1, *volume), spec["sigma"])
        n = (n / gain ** 0.5).reshape(2, *shape)
    elif kind == "heavy":
        df = spec["df"]
        chi2 = draw(df, 2, *shape).square().sum(0)
        n = draw(2, *shape) / torch.sqrt(chi2 / df) / np.sqrt(df / (df - 2))
    else:  # hetero: smooth log-gain map, power normalised to one over the volume
        coarse = draw(1, 1, 4, 4, 4)
        coarse = (coarse - coarse.mean()) / coarse.std().clamp_min(1e-6)
        gain = torch.exp(spec["depth"] * F.interpolate(coarse, size=volume, mode="trilinear", align_corners=False)[0, 0])
        n = draw(2, *shape) * (gain / gain.square().mean().sqrt())
    return torch.complex(n[0], n[1])


def per_ppm_per_s(b0: float) -> float:
    """Radians per ppm and per second."""
    return 2.0 * np.pi * GAMMA_BAR * b0


def estimate_ppm(signal: Tensor, tes, b0: float) -> Tensor:
    """Field in ppm from complex echoes [E, X, Y, Z]: phase unwrapped along the echoes, weighted (|S|^2) least-squares line with intercept."""
    t = torch.as_tensor(tes, device=signal.device, dtype=signal.real.dtype).reshape(-1, 1, 1, 1)
    phase = torch.angle(signal)
    step = phase[1:] - phase[:-1]
    step = step - 2 * np.pi * torch.round(step / (2 * np.pi))
    phase = torch.cat([phase[:1], phase[:1] + torch.cumsum(step, 0)], 0)
    w = signal.abs().square() + 1e-12
    wsum = w.sum(0)
    tm = (w * t).sum(0) / wsum
    pm = (w * phase).sum(0) / wsum
    slope = (w * (t - tm) * (phase - pm)).sum(0) / (w * (t - tm).square()).sum(0).clamp_min(1e-30)
    return slope / per_ppm_per_s(b0)


def weight_prime(magnitude_e: Tensor, tes) -> Tensor:
    """eNDI weight sum_e(m_e^2 TE_e) / sum_e(m_e TE_e) of the echo magnitudes [E, ...] (0 where undefined)."""
    t = torch.as_tensor(tes, device=magnitude_e.device, dtype=magnitude_e.dtype).reshape(-1, *([1] * (magnitude_e.ndim - 1)))
    numerator = (magnitude_e.square() * t).sum(0)
    denominator = (magnitude_e * t).sum(0)
    return torch.where(denominator > 0, numerator / denominator.clamp_min(1e-30), torch.zeros_like(numerator))
