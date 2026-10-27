"""Fresh dirty inputs from a clean field: the complex-noise recipe of the notebook (cell 6)."""

import torch
from torch import Tensor


def simulate_dirty(clean_ppm: Tensor, magnitude: Tensor, snr: float, phs_scale: float,
                   generator: torch.Generator | None = None, noise: Tensor | None = None) -> Tensor:
    """signal = mag exp(i b phs) + (1/snr)(randn + i randn); returns angle(signal)/phs in ppm.

    `noise`: a complex tensor of unit variance per component replacing the iid Gaussian draw (other noise statistics, CLAUDE.md 9bp).
    """
    if noise is None:
        draw = torch.randn((2, *clean_ppm.shape), generator=generator, device=clean_ppm.device, dtype=clean_ppm.dtype) / snr
        noise = torch.complex(draw[0], draw[1])
    else:
        noise = noise / snr
    signal = magnitude * torch.exp(1j * clean_ppm * phs_scale) + noise
    return torch.angle(signal) / phs_scale


def calibrate_snr(clean_ppm: Tensor, magnitude: Tensor, mask: Tensor, phs_scale: float, target_rms: float,
                  low=1.0, high=1000.0, iterations=14, seed=0) -> float:
    """SNR whose masked residual rms matches `target_rms` (bisection in log SNR; rms falls as SNR grows)."""
    generator = torch.Generator(device=clean_ppm.device).manual_seed(seed)
    for _ in range(iterations):
        snr = (low * high) ** 0.5
        residual = simulate_dirty(clean_ppm, magnitude, snr, phs_scale, generator) - clean_ppm
        if residual[mask > 0].square().mean().sqrt() > target_rms:
            low = snr
        else:
            high = snr
    return float((low * high) ** 0.5)
