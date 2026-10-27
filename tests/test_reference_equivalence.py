"""The torch dipolelets and kernel against the user's NumPy reference vendored in external/."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from mcno_qsm.dipolelets import SIM2_DIPOLELET, DipoleletTransform, band_counts, gaussian_smooth
from mcno_qsm.physics import dipole_kernel

EXTERNAL = Path(__file__).resolve().parents[1] / "external"
pytestmark = pytest.mark.skipif(not (EXTERNAL / "Dipolelet.py").exists(), reason="external/ not present")


@pytest.fixture(scope="module")
def reference():
    sys.path.insert(0, str(EXTERNAL))
    import Dipolelet
    import FANSI_tools
    import Starlet

    return Dipolelet, FANSI_tools, Starlet


def segments_as_images(transform, volume):
    x = torch.from_numpy(volume)[None, None]
    return [torch.fft.ifftn(s, dim=(-3, -2, -1), norm="ortho").real[0, 0].numpy() for s in transform.analysis(x)]


def reference_images(reference, volume, voxel_size, **kwargs):
    dipolelet, _, starlet = reference
    # The correctly aligned kernel, centered; see test_fansi_kernel_is_misaligned_for_odd_sizes.
    kernel = dipole_kernel(volume.shape, voxel_size, dtype=torch.float64).numpy()
    coeffs, coarse = dipolelet.parabolic_dipolelet_decomposition(
        image_3d=volume, D_qsm=np.fft.fftshift(kernel), transform=starlet.starlet_transform_3d, **kwargs)
    return [image for scale in coeffs for image in scale.values()] + list(coarse.values())


@pytest.mark.parametrize("shape,voxel", [((24, 28, 26), (1.0, 1.0, 1.0)), ((21, 25, 23), (0.9, 1.1, 1.4))])
def test_sim2_parameters_match_numpy_reference(reference, shape, voxel):
    volume = np.random.default_rng(0).standard_normal(shape)
    transform = DipoleletTransform(shape, voxel, dtype=torch.float64, **SIM2_DIPOLELET)
    expected = reference_images(
        reference, volume, voxel, num_radial_scales=3, base_angular_bands=3, delta0=0.05,
        starlet_filter_type="gaussian", starlet_filter_param=0.5, window_type="gaussian",
        gaussian_width_factor=0.5, reverse_scaling=False, scaling_law="static")
    got = segments_as_images(transform, volume)
    assert len(got) == len(expected) == 10
    for a, b in zip(got, expected):
        np.testing.assert_allclose(a, b, atol=1e-9)
    # Synthesis is the plain sum of the components.
    np.testing.assert_allclose(sum(got), volume, atol=1e-9)


@pytest.mark.parametrize("law,reverse", [("dyadic", False), ("curvelet", True)])
def test_other_laws_and_sigmoid_default_tightness_match_reference(reference, law, reverse):
    shape, voxel = (20, 22, 24), (1.0, 1.0, 1.0)
    volume = np.random.default_rng(1).standard_normal(shape)
    transform = DipoleletTransform(
        shape, voxel, dtype=torch.float64, scales=3, scaling_law=law, base_angular_bands=2,
        reverse_scaling=reverse, delta=0.05, tightness=None, window="sigmoid",
        starlet="gaussian", starlet_sigma=0.5)
    expected = reference_images(
        reference, volume, voxel, num_radial_scales=3, base_angular_bands=2, delta0=0.05,
        starlet_filter_type="gaussian", starlet_filter_param=0.5, window_type="sigmoid",
        tightness_factor=None, reverse_scaling=reverse, scaling_law=law)
    got = segments_as_images(transform, volume)
    assert len(got) == len(expected)
    for a, b in zip(got, expected):
        np.testing.assert_allclose(a, b, atol=1e-9)


def test_gaussian_smooth_matches_scipy(reference):
    from scipy.ndimage import gaussian_filter

    volume = np.random.default_rng(2).standard_normal((17, 19, 15))
    for sigma in (0.5, 1.0, 2.0):
        got = gaussian_smooth(torch.from_numpy(volume)[None, None], sigma)[0, 0].numpy()
        np.testing.assert_allclose(got, gaussian_filter(volume, sigma, mode="mirror"), atol=1e-12)


def test_band_counts_follow_reference_laws():
    assert band_counts(3, 3, "static") == (3, 3, 3)
    assert band_counts(3, 3, "dyadic") == (12, 6, 3)
    assert band_counts(3, 3, "dyadic", reverse=True) == (3, 6, 12)
    assert band_counts(3, 4, "curvelet") == (8, 6, 4)


@pytest.mark.parametrize("shape,voxel", [((16, 16, 16), (1, 1, 1)), ((16, 20, 24), (0.9, 1.1, 1.4)),
                                         ((164, 206, 206), (1.0, 1.0, 1.0))])
def test_repo_kernel_equals_fansi_kernel_for_even_sizes(reference, shape, voxel):
    _, tools, _ = reference
    fansi = tools.dipole_kernel(shape, np.asarray(shape) * np.asarray(voxel))
    ours = dipole_kernel(shape, voxel, dtype=torch.float64).numpy()
    np.testing.assert_allclose(ours, fansi, atol=1e-12)


@pytest.mark.parametrize("shape,voxel", [((15, 17, 19), (0.9, 1.1, 1.4)), ((164, 205, 205), (1.0, 1.0, 1.0))])
def test_fansi_kernel_is_misaligned_for_odd_sizes(reference, shape, voxel):
    # FANSI_tools.dipole_kernel returns fftshift(centered) where ifftshift is needed: the DC zero
    # lands off index 0 on odd axes. Ours has D(0)=0 at [0,0,0] and equals ifftshift(centered).
    _, tools, _ = reference
    fansi = tools.dipole_kernel(shape, np.asarray(shape) * np.asarray(voxel))
    ours = dipole_kernel(shape, voxel, dtype=torch.float64).numpy()
    assert ours[0, 0, 0] == 0 and fansi[0, 0, 0] != 0
    centered = np.fft.ifftshift(fansi)  # undo FANSI's final fftshift
    np.testing.assert_allclose(ours, np.fft.ifftshift(centered), atol=1e-12)
