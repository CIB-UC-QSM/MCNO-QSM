"""Differentiable dipole convolution in physical frequency coordinates."""

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def conjugate_partner(x: Tensor) -> Tensor:
    """Return x(-k) on the unshifted discrete Fourier grid (including Nyquist)."""
    for axis in range(x.ndim - 3, x.ndim):
        indices = (-torch.arange(x.shape[axis], device=x.device)) % x.shape[axis]
        x = x.index_select(axis, indices)
    return x


def hermitian_projection(spectrum: Tensor) -> Tensor:
    """Orthogonal projection onto spectra of real volumes."""
    return (spectrum + conjugate_partner(spectrum).conj()) * 0.5


def dipole_kernel(
    shape: tuple[int, int, int],
    voxel_size: tuple[float, float, float],
    b0: tuple[float, float, float] = (0.0, 0.0, 1.0),
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """D(k)=1/3-(k·b0)^2/|k|², D(0)=0, in unshifted FFT order.

    At mixed Nyquist planes an oblique sampled kernel may not be even.
    Symmetrizing the conjugate pairs makes real-valued convolution explicit.
    """
    if len(shape) != 3 or any(n < 2 for n in shape):
        raise ValueError("shape must contain three dimensions >= 2")
    if len(voxel_size) != 3 or any(not math.isfinite(v) or v <= 0 for v in voxel_size):
        raise ValueError("voxel_size must contain three finite positive spacings")
    if len(b0) != 3 or not all(math.isfinite(v) for v in b0) or math.hypot(*b0) == 0:
        raise ValueError("b0 must be a finite nonzero 3-vector")
    direction = torch.tensor(b0, device=device, dtype=dtype)
    direction = direction / direction.norm()
    frequencies = [torch.fft.fftfreq(n, d=v, device=device, dtype=dtype)
                   for n, v in zip(shape, voxel_size)]
    grid = torch.meshgrid(*frequencies, indexing="ij")
    radius2 = sum(k.square() for k in grid)
    projection = sum(k * d for k, d in zip(grid, direction))
    kernel = 1 / 3 - projection.square() / radius2.clamp_min(torch.finfo(dtype).tiny)
    kernel[0, 0, 0] = 0
    return (kernel + conjugate_partner(kernel)) * 0.5


class DipoleForward(nn.Module):
    """Zero-pad, convolve with the dipole, then crop to the input geometry."""

    def __init__(self, shape, voxel_size, b0=(0.0, 0.0, 1.0), padding=16):
        super().__init__()
        if not isinstance(padding, int) or padding < 0:
            raise ValueError("padding must be a nonnegative integer")
        self.shape = tuple(shape)
        self.padding = padding
        padded = tuple(n + 2 * padding for n in shape)
        self.register_buffer("kernel", dipole_kernel(padded, tuple(voxel_size), tuple(b0)))

    def forward(self, susceptibility: Tensor) -> Tensor:
        if susceptibility.shape[-3:] != self.shape:
            raise ValueError("susceptibility and forward geometry do not match")
        p = self.padding
        padded = F.pad(susceptibility, (p,) * 6) if p else susceptibility
        field = torch.fft.ifftn(
            torch.fft.fftn(padded, dim=(-3, -2, -1), norm="ortho") * self.kernel,
            dim=(-3, -2, -1), norm="ortho",
        ).real
        return field[..., p:p + self.shape[0], p:p + self.shape[1], p:p + self.shape[2]]
