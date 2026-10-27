import torch

from mcno_qsm.losses import BAND_D_EDGES, BAND_K_EDGES, band_loss, band_masks, phase_loss

SHAPE, VOXEL = (24, 20, 16), (1.0, 1.1, 0.9)


def test_band_masks_partition_k_space_twice():
    bands = band_masks(SHAPE, VOXEL)
    n_d = len(BAND_D_EDGES)
    assert bands.shape == (n_d + len(BAND_K_EDGES), *SHAPE)
    assert torch.equal(bands[:n_d].sum(0), torch.ones(SHAPE))   # |D| bands: every frequency in exactly one
    assert torch.equal(bands[n_d:].sum(0), torch.ones(SHAPE))   # |k| bands too


def test_band_loss_is_zero_for_a_perfect_prediction_and_positive_otherwise():
    torch.manual_seed(0)
    bands = band_masks(SHAPE, VOXEL)
    target = torch.randn(2, 1, *SHAPE)
    mask = torch.ones_like(target)
    zero, per_band = band_loss(target, target, mask, bands)
    assert zero < 1e-6 and per_band.shape == (bands.shape[0],)
    assert band_loss(target + 0.1 * torch.randn_like(target), target, mask, bands)[0] > 0.01


def test_removing_cone_content_costs_more_than_equal_energy_high_frequency_error():
    """Same image-space error energy: dropping the (weak) cone band of the truth vs adding noise in the strongest band."""
    torch.manual_seed(1)
    bands = band_masks(SHAPE, VOXEL)
    cone = bands[0].bool()                                   # |D| < 0.03
    spectrum = torch.fft.fftn(torch.randn(1, 1, *SHAPE), dim=(-3, -2, -1), norm="ortho")
    # truth: strong outside the cone, weak inside it
    truth = torch.fft.ifftn(spectrum * torch.where(cone, 0.1, 1.0), dim=(-3, -2, -1), norm="ortho").real
    mask = torch.ones_like(truth)
    lost = truth - torch.fft.ifftn(torch.fft.fftn(truth, dim=(-3, -2, -1), norm="ortho") * ~cone, dim=(-3, -2, -1), norm="ortho").real
    noise = torch.randn_like(truth)
    noise = noise / noise.norm() * lost.norm()               # identical error energy
    removed_cone = band_loss(truth - lost, truth, mask, bands)[0]
    added_noise = band_loss(truth + noise, truth, mask, bands)[0]
    assert abs(phase_loss(truth - lost, truth, mask)[2].item() - phase_loss(truth + noise, truth, mask)[2].item()) < 1e-5
    assert removed_cone > added_noise


def test_band_loss_gradient_flows_and_is_finite():
    bands = band_masks(SHAPE, VOXEL)
    target = torch.randn(1, 1, *SHAPE)
    prediction = (target + 0.2 * torch.randn_like(target)).requires_grad_()
    loss, _ = band_loss(prediction, target, torch.ones_like(target), bands)
    loss.backward()
    assert torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum() > 0
