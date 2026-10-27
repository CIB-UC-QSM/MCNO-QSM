"""Masked phase losses, the dipole-cone energy diagnostic and the band-normalized spectral loss."""

from torch import Tensor
import torch

from .physics import dipole_kernel


def phase_loss(prediction: Tensor, target: Tensor, mask: Tensor, l1_weight=1.0, l2_weight=1.0):
    """Masked L1 + L2, each normalized by the norm of the masked target. Returns (loss, l1, l2)."""
    mask = mask.to(prediction.dtype)
    difference = (prediction - target) * mask
    reference = target * mask
    l2 = difference.norm() / reference.norm().clamp_min(1e-12)
    l1 = difference.abs().sum() / reference.abs().sum().clamp_min(1e-12)
    return l1_weight * l1 + l2_weight * l2, l1, l2


def cone_mask(shape, voxel_size, delta, b0=(0.0, 0.0, 1.0), device=None) -> Tensor:
    return dipole_kernel(shape, voxel_size, b0, device=device).abs() < delta


def cone_energy(x: Tensor, cone: Tensor) -> Tensor:
    """Share of the spectral energy of x inside |D(k)| < delta."""
    power = torch.fft.fftn(x, dim=(-3, -2, -1), norm="ortho").abs().square()
    return (power * cone).sum() / power.sum().clamp_min(1e-30)


def cone_loss(prediction: Tensor, target: Tensor, cone: Tensor) -> Tensor:
    """Excess cone-energy share of the prediction over that of the target (zero if it has no more)."""
    return torch.relu(cone_energy(prediction, cone) - cone_energy(target, cone))


# |D| bands (the cone and its surroundings) and |k| bands (cycles/mm); each family partitions k-space, the last band is open.
BAND_D_EDGES = (0.0, 0.03, 0.15, 0.3)
BAND_K_EDGES = (0.0, 0.05, 0.1, 0.2, 0.3)


def band_masks(shape, voxel_size, b0=(0.0, 0.0, 1.0), device=None) -> Tensor:
    """[n_bands,X,Y,Z] float masks: |D| bands first, then |k| bands, in the native FFT order of `shape`."""
    kernel = dipole_kernel(shape, voxel_size, b0, device=device).abs()
    freqs = [torch.fft.fftfreq(n, d=v, device=device) for n, v in zip(shape, voxel_size)]
    radius = sum(k.square() for k in torch.meshgrid(*freqs, indexing="ij")).sqrt()
    bands = []
    for values, edges in ((kernel, BAND_D_EDGES), (radius, BAND_K_EDGES)):
        bounds = [*edges, float("inf")]
        bands += [(values >= lo) & (values < hi) for lo, hi in zip(bounds[:-1], bounds[1:])]
    return torch.stack(bands).to(kernel.dtype)


def band_loss(prediction: Tensor, target: Tensor, mask: Tensor, bands: Tensor, floor=1e-3):
    """Mean over bands of the relative error ||E_b|| / sqrt(||C_b||^2 + floor ||C||^2).

    E and C are the orthonormal spectra of the masked error and the masked target; the floor (a share of the total
    target power) keeps bands with almost no true signal, such as the cone, from dividing by ~0. Every band counts
    equally, so removing true signal in the weak cone bands costs as much as in the strong ones (the image-space
    losses are dominated by the strong bands). Returns (loss, per-band relative errors averaged over the batch).
    """
    mask = mask.to(prediction.dtype)
    error = torch.fft.fftn((prediction - target) * mask, dim=(-3, -2, -1), norm="ortho").abs().square().squeeze(1)
    truth = torch.fft.fftn(target * mask, dim=(-3, -2, -1), norm="ortho").abs().square().squeeze(1)
    error_band = torch.einsum("bxyz,nxyz->bn", error, bands)
    truth_band = torch.einsum("bxyz,nxyz->bn", truth, bands)
    relative = (error_band / (truth_band + floor * truth.sum(dim=(-3, -2, -1))[:, None]).clamp_min(1e-30) + 1e-30).sqrt()
    return relative.mean(), relative.mean(dim=0).detach()


def segment_loss(prediction: Tensor, target: Tensor, mask: Tensor, transform, gains) -> Tensor:
    """sqrt(sum_s g_s^2 ||M e_s||^2) / ||M b||, e_s = image-domain component of the masked error in Dipole-let segment s.

    `transform` is a DipoleletTransform for the patch geometry; `gains` has one sensitivity per segment (CLAUDE.md 9ai/9aj: the
    eNDI response to an error in that segment). The components sum exactly to the error but overlap, so with all gains equal
    this is NOT the plain L2 norm; it is added to the usual loss, not substituted for it.
    """
    mask = mask.to(prediction.dtype)
    error = (prediction - target) * mask
    gains = [float(g) for g in gains]
    if len(gains) != len(transform.segment_names):
        raise ValueError(f"{len(gains)} gains for {len(transform.segment_names)} segments")
    total = prediction.new_zeros(())
    for gain, spectrum in zip(gains, transform.analysis(error)):
        component = torch.fft.ifftn(spectrum, dim=(-3, -2, -1), norm="ortho").real * mask
        total = total + gain**2 * component.square().sum()
    return (total + 1e-24).sqrt() / (target * mask).norm().clamp_min(1e-12)
