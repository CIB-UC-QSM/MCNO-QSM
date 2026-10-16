"""Starlet B3 details and angular windows, adapted from ignacontreras/Dipolelets.

Reference: Codes/Starlet.py and Codes/Dipolelet.py at cf98d579.
Unlike the reference, analysis returns complex Fourier components, not images.
"""

from collections.abc import Iterable, Iterator

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .physics import dipole_kernel


def create_dipolelet_windows(
    kernel: Tensor, thresholds: Iterable[float], tightness: float = 100.0,
) -> Tensor:
    """Return [bands,X,Y,Z] smooth windows ordered core, annuli, outer.

    Thresholds are strictly increasing positive values of |D(k)|. Adjacent
    sigmoid disks define annuli; the normalized windows form a partition of
    unity. Empty thresholds produce one all-pass window. Windows overlap.
    """
    thresholds = tuple(thresholds)
    if not 0 < tightness < float("inf"):
        raise ValueError("tightness must be finite and positive")
    if any(not 0 < t < float("inf") for t in thresholds):
        raise ValueError("thresholds must be finite and positive")
    if any(a >= b for a, b in zip(thresholds, thresholds[1:])):
        raise ValueError("thresholds must be strictly increasing")
    if not thresholds:
        return torch.ones_like(kernel).unsqueeze(0)
    disks = [torch.sigmoid(tightness * (t - kernel.abs())) for t in thresholds]
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


class DipoleletTransform(nn.Module):
    """Lazy, differentiable analysis of [B,C,X,Y,Z] into Fourier segments.

    Three default detail scales use 8,4,2 angular bands. The final coarse
    component is unsegmented. Synthesis is a sum, with no second windowing.
    """

    def __init__(self, shape, voxel_size, b0=(0.0, 0.0, 1.0), *,
                 scales=3, angular_bands=(8, 4, 2), delta=0.03, tightness=100.0):
        super().__init__()
        if scales < 1 or len(angular_bands) != scales or any(n < 1 for n in angular_bands):
            raise ValueError("provide one positive band count per detail scale")
        if not 0 < delta < 1 / 3:
            raise ValueError("delta must be between 0 and 1/3")
        self.shape = tuple(shape)
        self.scales = scales
        self.segment_names = []
        kernel = dipole_kernel(self.shape, tuple(voxel_size), tuple(b0))
        for j, bands in enumerate(angular_bands):
            # Match np.logspace(delta, 1/3, bands-1), including its one-threshold case.
            thresholds = torch.logspace(torch.log10(torch.tensor(delta)).item(),
                                       torch.log10(torch.tensor(1 / 3)).item(), bands - 1).tolist()
            windows = create_dipolelet_windows(kernel, thresholds, tightness)
            self.register_buffer(f"windows_{j}", windows, persistent=False)
            self.segment_names.extend(f"scale{j + 1}/band{k}" for k in range(bands))
        self.segment_names.append("coarse")

    def analysis(self, x: Tensor) -> Iterator[Tensor]:
        if x.ndim != 5 or x.shape[-3:] != self.shape:
            raise ValueError("expected [B,C,X,Y,Z] matching the transform geometry")
        coarse = x
        for j in range(self.scales):
            smoothed = starlet_smooth(coarse, 2**j)
            spectrum = torch.fft.fftn(coarse - smoothed, dim=(-3, -2, -1), norm="ortho")
            for window in getattr(self, f"windows_{j}"):
                yield spectrum * window
            coarse = smoothed
        yield torch.fft.fftn(coarse, dim=(-3, -2, -1), norm="ortho")

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
