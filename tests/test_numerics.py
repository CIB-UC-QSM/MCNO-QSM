import copy

import numpy as np
import pytest
import torch
from scipy.ndimage import convolve1d

from mcno_qsm.dipolelets import DipoleletTransform, create_dipolelet_windows, starlet_smooth
from mcno_qsm.metrics import QSMObjective, masked_nrmse
from mcno_qsm.model import ComplexMLP, DipoleletNet, StackedDipoleletNet, build_model
from mcno_qsm.physics import DipoleForward, conjugate_partner, dipole_kernel, hermitian_projection


@pytest.fixture(autouse=True)
def deterministic_small_cpu():
    torch.set_num_threads(2)
    torch.manual_seed(42)


@pytest.mark.parametrize("shape", [(8, 10, 12), (7, 9, 11)])
def test_windows_partition_and_inverse(shape):
    transform = DipoleletTransform(shape, (0.9, 1.1, 1.4))
    assert len(transform.segment_names) == 15
    for j, bands in enumerate((8, 4, 2)):
        windows = getattr(transform, f"windows_{j}")
        assert windows.shape == (bands, *shape)
        assert windows.min() >= 0
        torch.testing.assert_close(windows.sum(0), torch.ones(shape))
    x = torch.randn(2, 6, *shape)
    segments = list(transform.analysis(x))
    assert len(segments) == 15 and all(s.is_complex() for s in segments)
    torch.testing.assert_close(transform.synthesis(segments), x, atol=2e-6, rtol=2e-6)


def test_windows_match_reference_sigmoid_formula():
    kernel = torch.linspace(-2 / 3, 1 / 3, 127, dtype=torch.float64)
    thresholds = [0.03, 0.1, 1 / 3]
    disks = [1 / (1 + np.exp(100 * (np.abs(kernel.numpy()) - t))) for t in thresholds]
    expected = np.stack([disks[0], disks[1] - disks[0], disks[2] - disks[1], 1 - disks[2]])
    expected /= expected.sum(axis=0)
    np.testing.assert_allclose(create_dipolelet_windows(kernel, thresholds).numpy(), expected,
                               rtol=1e-12, atol=1e-15)
    torch.testing.assert_close(create_dipolelet_windows(kernel, []), torch.ones_like(kernel)[None])
    with pytest.raises(ValueError):
        create_dipolelet_windows(kernel, [0.1, 0.03])


@pytest.mark.parametrize("step", [1, 2, 4])
def test_starlet_matches_scipy_mirror(step):
    x = torch.randn(1, 2, 7, 9, 11, dtype=torch.float64)
    kernel = np.zeros(4 * step + 1)
    kernel[::step] = np.array([1, 4, 6, 4, 1]) / 16
    expected = x.numpy()
    for axis in (2, 3, 4):
        expected = convolve1d(expected, kernel, axis=axis, mode="mirror")
    np.testing.assert_allclose(starlet_smooth(x, step).numpy(), expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("shape", [(8, 10, 12), (7, 9, 11)])
def test_kernel_and_hermitian_projection(shape):
    kernel = dipole_kernel(shape, (0.8, 1.2, 1.7), b0=(0.2, -0.3, 0.9))
    assert kernel[0, 0, 0] == 0
    torch.testing.assert_close(kernel, conjugate_partner(kernel))
    spectrum = hermitian_projection(torch.randn(*shape, dtype=torch.complex64))
    torch.testing.assert_close(spectrum, conjugate_partner(spectrum).conj())
    assert torch.fft.ifftn(spectrum).imag.abs().max() < 1e-6
    axis_kernel = dipole_kernel(shape, (1, 1, 1))
    assert axis_kernel[1, 0, 0].item() == pytest.approx(1 / 3)
    assert axis_kernel[0, 0, 1].item() == pytest.approx(-2 / 3)


def test_forward_known_signals_and_padding():
    shape = (8, 10, 12)
    physics = DipoleForward(shape, (1, 1, 1), padding=0)
    torch.testing.assert_close(physics(torch.ones(1, 1, *shape)), torch.zeros(1, 1, *shape))
    z = torch.arange(shape[2])
    wave = torch.cos(2 * torch.pi * z / shape[2]).expand(1, 1, *shape)
    torch.testing.assert_close(physics(wave), -2 / 3 * wave, atol=2e-7, rtol=2e-6)
    padded = DipoleForward(shape, (1, 1, 1), padding=3)
    x = torch.randn(1, 1, *shape, requires_grad=True)
    y = padded(x)
    reference = torch.fft.ifftn(torch.fft.fftn(torch.nn.functional.pad(x, (3,) * 6),
                                               dim=(-3, -2, -1)) * padded.kernel,
                                dim=(-3, -2, -1)).real[..., 3:11, 3:13, 3:15]
    torch.testing.assert_close(y, reference)
    y.square().sum().backward()
    assert torch.isfinite(x.grad).all() and x.grad.norm() > 0


def test_masked_nrmse_exact():
    gt = torch.tensor([3.0, 4.0, 999.0])
    mask = torch.tensor([True, True, False])
    pred = torch.tensor([0.0, 0.0, -999.0])
    assert masked_nrmse(pred, gt, mask) == 1
    assert masked_nrmse(gt, gt, mask) == 0
    assert masked_nrmse(2 * gt, gt, mask) == 1
    with pytest.raises(ValueError):
        masked_nrmse(gt, gt * 0, mask)


def test_objective_gt_zero_and_normalization():
    shape = (7, 8, 9)
    mask = torch.ones(1, 1, *shape, dtype=torch.bool)
    mask[..., 0, :, :] = False
    gt = torch.randn(1, 1, *shape) * 0.03
    objective = QSMObjective(DipoleForward(shape, (1, 1, 1), padding=2), gt, mask)
    total, physical, l1 = objective(gt)
    assert total == physical == l1 == 0
    total, physical, l1 = objective(torch.zeros_like(gt))
    torch.testing.assert_close(physical, torch.tensor(1.0))
    torch.testing.assert_close(l1, torch.tensor(1.0))
    torch.testing.assert_close(total, torch.tensor(2.0))


def test_model_shapes_isolation_chunks_and_gradients():
    shape = (7, 8, 9)
    model = DipoleletNet(shape, (1, 1, 1), scales=1, angular_bands=[2],
                        width=4, chunk_size=128, checkpoint=True)
    assert [tuple(c.kernel_size) for c in model.convolutions] == [(3, 3, 3), (5, 5, 5), (7, 7, 7)]
    assert all(c.out_channels == 2 for c in model.convolutions)
    assert len(model.mlps) == 3 and all(len(m.layers) == 10 for m in model.mlps)
    assert not (set(map(id, model.mlps[0].parameters())) & set(map(id, model.mlps[1].parameters())))
    segment = torch.randn(2, 6, *shape, dtype=torch.complex64)
    output = model.process_segment(segment, model.mlps[0])
    expected = model.mlps[0](segment.movedim(1, -1)).squeeze(-1).unsqueeze(1)
    torch.testing.assert_close(output, expected, atol=1e-6, rtol=1e-5)
    reference = copy.deepcopy(model)
    reference.use_checkpoint = False
    field = torch.randn(1, 1, *shape)
    assert model.cnn(field).shape == (1, 6, *shape)
    result = model(field)
    assert result.shape == field.shape and not result.is_complex()
    torch.testing.assert_close(result, reference(field))
    result.square().mean().backward()
    reference(field).square().mean().backward()
    for p, q in zip(model.parameters(), reference.parameters()):
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
        torch.testing.assert_close(p.grad, q.grad, atol=1e-7, rtol=1e-4)


def test_small_overfit_reduces_loss():
    shape = (8, 8, 8)
    grid = torch.meshgrid(*[torch.linspace(-1, 1, n) for n in shape], indexing="ij")
    gt = (0.08 * torch.exp(-sum(x.square() for x in grid) / 0.2))[None, None]
    mask = torch.ones_like(gt, dtype=torch.bool)
    physics = DipoleForward(shape, (1, 1, 1), padding=2)
    objective = QSMObjective(physics, gt, mask)
    field = physics(gt)
    field = field / field.std()
    model = DipoleletNet(shape, (1, 1, 1), scales=1, angular_bands=[2], width=8, checkpoint=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    initial = objective(model(field))[0].item()
    for _ in range(40):
        optimizer.zero_grad()
        loss = objective(model(field))[0]
        loss.backward()
        optimizer.step()
    final = objective(model(field))[0].item()
    assert final < 0.8 * initial, (initial, final)


def test_stacking_preserves_baseline_and_composes_independent_layers():
    shape = (6, 7, 8)
    kwargs = dict(scales=1, angular_bands=[2], width=4, checkpoint=True, chunk_size=128)
    torch.manual_seed(42)
    original = DipoleletNet(shape, (1, 1, 1), **kwargs)
    torch.manual_seed(42)
    single = build_model(shape, (1, 1, 1), depth=1, **kwargs)
    assert type(single) is DipoleletNet
    for key, value in original.state_dict().items():
        torch.testing.assert_close(value, single.state_dict()[key], rtol=0, atol=0)
    stack = build_model(shape, (1, 1, 1), depth=3, **kwargs)
    assert isinstance(stack, StackedDipoleletNet) and len(stack.layers) == 3
    parameters = [set(map(id, layer.parameters())) for layer in stack.layers]
    assert not (parameters[0] & parameters[1] or parameters[1] & parameters[2])
    field = torch.randn(1, 1, *shape)
    result = stack(field)
    expected = stack.layers[2](stack.layers[1](stack.layers[0](field)))
    torch.testing.assert_close(result, expected)
    result.square().mean().backward()
    for layer in stack.layers:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in layer.parameters())
        assert sum(p.grad.abs().sum() for p in layer.parameters()) > 0
    with pytest.raises(ValueError):
        build_model(shape, (1, 1, 1), depth=0)
