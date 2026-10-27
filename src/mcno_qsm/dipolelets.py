"""Starlet details and angular windows, adapted from ignacontreras/Dipolelets.

Reference: Codes/Starlet.py and Codes/Dipolelet.py at cf98d579.
Unlike the reference, analysis returns complex Fourier components, not images.
The reference options (gaussian starlet, erf windows, 20/width sigmoid, scaling laws) are selectable;
the defaults keep the original B3 / sigmoid(100) / 8,4,2 behaviour.
"""

import math
from collections.abc import Iterable, Iterator

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .physics import dipole_kernel

# Parameters of the Sim2 experiments (utils/Dipole-lets_Paper_Plots_3.ipynb, cell 19).
SIM2_DIPOLELET = dict(
    scales=3, scaling_law="static", base_angular_bands=3, delta=0.05, window="gaussian",
    gaussian_width_factor=0.5, starlet="gaussian", starlet_sigma=0.5,
)


def band_counts(scales: int, base: int, law: str = "static", reverse: bool = False) -> tuple[int, ...]:
    """Angular bands per detail scale (Dipolelet.py: static, dyadic, curvelet)."""
    if law not in ("static", "dyadic", "curvelet"):
        raise ValueError("scaling_law must be 'static', 'dyadic' or 'curvelet'")
    counts = []
    for j in range(scales):
        if law == "static":
            counts.append(base)
            continue
        exponent = j if reverse else scales - 1 - j
        factor = 2**exponent if law == "dyadic" else 2 ** (exponent / 2.0)
        counts.append(max(1, int(np.round(base * factor))))
    return tuple(counts)


def _transition_widths(thresholds: tuple[float, ...]) -> list[float]:
    # Width of the transition around each threshold from its neighbours, as in the reference.
    if len(thresholds) == 1:
        return [thresholds[0]]
    return [thresholds[1] - thresholds[0], *[b - a for a, b in zip(thresholds, thresholds[1:])]]


def create_dipolelet_windows(
    kernel: Tensor, thresholds: Iterable[float], tightness: float | None = 100.0, *,
    window: str = "sigmoid", width_factor: float = 1.0,
) -> Tensor:
    """Return [bands,X,Y,Z] smooth windows ordered core, annuli, outer.

    Thresholds are strictly increasing positive values of |D(k)|. Adjacent
    disks define annuli; the normalized windows form a partition of unity.
    Empty thresholds produce one all-pass window. Windows overlap.

    window="sigmoid": disk = sigmoid(tightness (t-|D|)); tightness=None uses 20/width per
    threshold like the reference. window="gaussian": disk = 1 - (1+erf((|D|-t)/(sqrt2 sigma)))/2
    with sigma = width_factor*width/2 (the error-function window of the reference).
    """
    thresholds = tuple(thresholds)
    if window not in ("sigmoid", "gaussian"):
        raise ValueError("window must be 'sigmoid' or 'gaussian'")
    if tightness is not None and not 0 < tightness < float("inf"):
        raise ValueError("tightness must be finite and positive")
    if not 0 < width_factor < float("inf"):
        raise ValueError("width_factor must be finite and positive")
    if any(not 0 < t < float("inf") for t in thresholds):
        raise ValueError("thresholds must be finite and positive")
    if any(a >= b for a, b in zip(thresholds, thresholds[1:])):
        raise ValueError("thresholds must be strictly increasing")
    if not thresholds:
        return torch.ones_like(kernel).unsqueeze(0)
    magnitude = kernel.abs()
    widths = _transition_widths(thresholds)
    if window == "sigmoid":
        slopes = [tightness if tightness is not None else 20.0 / (w + 1e-9) for w in widths]
        disks = [torch.sigmoid(k * (t - magnitude)) for k, t in zip(slopes, thresholds)]
    else:
        disks = [1.0 - 0.5 * (1 + torch.erf((magnitude - t) / (math.sqrt(2) * (w * width_factor / 2) + 1e-9)))
                 for t, w in zip(thresholds, widths)]
    windows = torch.stack([disks[0], *[(b - a).clamp_min(0)
                                     for a, b in zip(disks, disks[1:])], 1 - disks[-1]])
    return windows / windows.sum(dim=0).clamp_min(torch.finfo(kernel.dtype).tiny)


def _mirror_indices(length: int, padding: int, device: torch.device) -> Tensor:
    # scipy.ndimage mode='mirror': whole-sample symmetry, excluding endpoints.
    indices = torch.arange(-padding, length + padding, device=device) % (2 * (length - 1))
    return torch.minimum(indices, 2 * (length - 1) - indices)


def starlet_smooth(x: Tensor, step: int) -> Tensor:
    """Separable dilated B3 smoothing with whole-sample mirror boundaries."""
    channels = x.shape[1]
    base = x.new_tensor([1, 4, 6, 4, 1]) / 16
    for axis in range(3):
        kernel_shape = [1, 1, 1]
        kernel_shape[axis] = 5
        weight = base.reshape(1, 1, *kernel_shape).expand(channels, 1, *kernel_shape)
        dim = axis + 2
        padded = x.index_select(dim, _mirror_indices(x.shape[dim], 2 * step, x.device))
        dilation = [1, 1, 1]
        dilation[axis] = step
        x = F.conv3d(padded, weight, dilation=tuple(dilation), groups=channels)
    return x


def gaussian_smooth(x: Tensor, sigma: float, truncate: float = 4.0) -> Tensor:
    """Separable Gaussian (scipy.ndimage.gaussian_filter, mode='mirror', same truncation)."""
    channels = x.shape[1]
    radius = int(truncate * sigma + 0.5)
    t = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    base = torch.exp(-0.5 * (t / sigma) ** 2)
    base = base / base.sum()
    for axis in range(3):
        kernel_shape = [1, 1, 1]
        kernel_shape[axis] = 2 * radius + 1
        weight = base.reshape(1, 1, *kernel_shape).expand(channels, 1, *kernel_shape)
        dim = axis + 2
        x = F.conv3d(x.index_select(dim, _mirror_indices(x.shape[dim], radius, x.device)), weight,
                     groups=channels)
    return x


class DipoleletTransform(nn.Module):
    """Lazy, differentiable analysis of [B,C,X,Y,Z] into Fourier segments.

    Three default detail scales use 8,4,2 angular bands. The final coarse
    component is unsegmented. Synthesis is a sum, with no second windowing.
    `scaling_law` (+ `base_angular_bands`, `reverse_scaling`) replaces `angular_bands`;
    `starlet`="gaussian" uses sigma `starlet_sigma`*2^j instead of the dilated B3.
    """

    def __init__(self, shape, voxel_size, b0=(0.0, 0.0, 1.0), *,
                 scales=3, angular_bands=(8, 4, 2), delta=0.03, tightness=100.0,
                 window="sigmoid", gaussian_width_factor=1.0, starlet="b3", starlet_sigma=0.5,
                 scaling_law=None, base_angular_bands=3, reverse_scaling=False,
                 coarse_bands=1, dtype=torch.float32):
        super().__init__()
        if coarse_bands < 1:
            raise ValueError("coarse_bands must be >= 1")
        if scaling_law is not None:
            angular_bands = band_counts(scales, base_angular_bands, scaling_law, reverse_scaling)
        if starlet not in ("b3", "gaussian"):
            raise ValueError("starlet must be 'b3' or 'gaussian'")
        if not 0 < starlet_sigma < float("inf"):
            raise ValueError("starlet_sigma must be finite and positive")
        if scales < 1 or len(angular_bands) != scales or any(n < 1 for n in angular_bands):
            raise ValueError("provide one positive band count per detail scale")
        if not 0 < delta < 1 / 3:
            raise ValueError("delta must be between 0 and 1/3")
        self.shape = tuple(shape)
        self.scales = scales
        self.starlet = starlet
        self.starlet_sigma = float(starlet_sigma)
        self.segment_names = []
        kernel = dipole_kernel(self.shape, tuple(voxel_size), tuple(b0), dtype=dtype)
        for j, bands in enumerate(angular_bands):
            # np.logspace(delta, 1/3, bands-1), including its one-threshold case.
            thresholds = np.logspace(np.log10(delta), np.log10(1 / 3), bands - 1).tolist()
            windows = create_dipolelet_windows(kernel, thresholds, tightness, window=window,
                                               width_factor=gaussian_width_factor)
            self.register_buffer(f"windows_{j}", windows, persistent=False)
            self.segment_names.extend(f"scale{j + 1}/band{k}" for k in range(bands))
        self.coarse_bands = int(coarse_bands)
        if self.coarse_bands > 1:
            # CLAUDE.md 9ap/9aq: the coarse remainder split by |D| with the same thresholds as the detail scales (core, annuli, outer).
            thresholds = np.logspace(np.log10(delta), np.log10(1 / 3), self.coarse_bands - 1).tolist()
            self.register_buffer("windows_coarse", create_dipolelet_windows(
                kernel, thresholds, tightness, window=window, width_factor=gaussian_width_factor), persistent=False)
            self.segment_names.extend(f"coarse/band{k}" for k in range(self.coarse_bands))
        else:
            self.segment_names.append("coarse")

    def analysis_parts(self, x: Tensor) -> Iterator[tuple[Tensor, Tensor | None]]:
        """Yield (spectrum, window) per segment; segment = spectrum * window (window None = coarse).

        Segments of one scale share a spectrum, so a caller can recompute `spectrum * window`
        per segment (memory) instead of keeping every windowed spectrum alive.
        """
        if x.ndim != 5 or x.shape[-3:] != self.shape:
            raise ValueError("expected [B,C,X,Y,Z] matching the transform geometry")
        coarse = x
        for j in range(self.scales):
            smoothed = (starlet_smooth(coarse, 2**j) if self.starlet == "b3"
                        else gaussian_smooth(coarse, self.starlet_sigma * 2**j))
            spectrum = torch.fft.fftn(coarse - smoothed, dim=(-3, -2, -1), norm="ortho")
            for window in getattr(self, f"windows_{j}"):
                yield spectrum, window
            coarse = smoothed
        spectrum = torch.fft.fftn(coarse, dim=(-3, -2, -1), norm="ortho")
        if self.coarse_bands > 1:
            for window in self.windows_coarse:
                yield spectrum, window
        else:
            yield spectrum, None

    def analysis(self, x: Tensor) -> Iterator[Tensor]:
        for spectrum, window in self.analysis_parts(x):
            yield spectrum if window is None else spectrum * window

    @staticmethod
    def synthesis(segments: Iterable[Tensor]) -> Tensor:
        iterator = iter(segments)
        try:
            spectrum = next(iterator)
        except StopIteration as exc:
            raise ValueError("at least one segment is required") from exc
        for segment in iterator:
            spectrum = spectrum + segment
        return torch.fft.ifftn(spectrum, dim=(-3, -2, -1), norm="ortho").real


def image_components(transform: "DipoleletTransform", x: Tensor) -> list[Tensor]:
    """Image-domain component of every segment of `x` ([X,Y,Z]); they sum exactly to `x` (segments + coarse remainder)."""
    out = []
    for segment in transform.analysis(x[None, None]):
        out.append(torch.fft.ifftn(segment, dim=(-3, -2, -1), norm="ortho").real[0, 0])
    return out


class PlainTransform(nn.Module):
    """Ablation: a single segment holding the whole spectrum (a plain FNO block with one C x C mix)."""

    segment_names = ["full"]

    def __init__(self, shape, *args, **kwargs):
        super().__init__()
        self.shape = tuple(shape)

    def analysis_parts(self, x: Tensor) -> Iterator[tuple[Tensor, Tensor | None]]:
        if x.ndim != 5 or x.shape[-3:] != self.shape:
            raise ValueError("expected [B,C,X,Y,Z] matching the transform geometry")
        yield torch.fft.fftn(x, dim=(-3, -2, -1), norm="ortho"), None
