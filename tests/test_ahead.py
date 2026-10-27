import nibabel as nib
import numpy as np
import pytest
from scipy import ndimage

from mcno_qsm.ahead import b0_in_voxel_axes, find_subjects, preprocess_subject, resample_to_world, world_grid
from mcno_qsm.physics import dipole_kernel


def _affine(zooms, tilt_deg, flip_x):
    t = np.radians(tilt_deg)
    tilt = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
    rotation = tilt @ np.diag([-zooms[0] if flip_x else zooms[0], zooms[1], zooms[2]])
    affine = np.eye(4)
    affine[:3, :3] = rotation
    affine[:3, 3] = [40.0, -30.0, -20.0]
    return affine


def _world_volume(points):
    """Smooth analytic field of world coordinates (mm)."""
    x, y, z = points
    return np.exp(-((x - 5) ** 2 + (y + 3) ** 2 + (z - 8) ** 2) / (2 * 14.0**2)) + 0.3 * np.sin(x / 11.0) * np.cos(z / 13.0)


def _native(affine, shape):
    idx = np.indices(shape).reshape(3, -1).T.astype(float)
    world = idx @ affine[:3, :3].T + affine[:3, 3]
    return _world_volume(world.T).reshape(shape).astype(np.float32)


@pytest.mark.parametrize("tilt,flip", [(0.0, False), (6.6, True), (26.0, True)])
def test_resample_recovers_the_world_frame_volume(tilt, flip):
    affine = _affine((0.7, 0.64, 0.64), tilt, flip)
    shape = (70, 90, 90)
    native = _native(affine, shape)
    support = np.ones(shape, bool)
    origin, out_shape = world_grid(affine, shape, support, margin_mm=-8.0, minimum=8)
    out = resample_to_world(native, affine, origin, out_shape, antialias=False)
    grid = np.indices(out_shape).reshape(3, -1).T * 1.0 + origin
    expected = _world_volume(grid.T).reshape(out_shape)
    native_idx = (grid - affine[:3, 3]) @ np.linalg.inv(affine[:3, :3]).T          # the tilted box does not fill the world box
    inside = np.all((native_idx > 2) & (native_idx < np.array(shape) - 3), axis=1).reshape(out_shape)
    assert inside.mean() > 0.2
    assert np.abs(out - expected)[inside].max() < 5e-3


def test_b0_direction_in_voxel_axes_matches_the_tilt():
    b0, tilt = b0_in_voxel_axes(_affine((0.7, 0.64, 0.64), 12.0, True))
    assert tilt == pytest.approx(12.0, abs=1e-6)
    assert abs(b0[0]) < 1e-9 and abs(b0[2]) == pytest.approx(np.cos(np.radians(12.0)))


def test_dipole_field_is_consistent_after_resampling():
    """A tilted acquisition sampled from a world-frame field gives, after resampling, the world-frame field (B0 along z)."""
    shape = (64, 64, 64)
    kernel = dipole_kernel(shape, (1.0, 1.0, 1.0), dtype=__import__("torch").float64).numpy()
    chi = np.zeros(shape)
    chi[28:36, 28:36, 28:36] = 1.0
    chi = ndimage.gaussian_filter(chi, 2.0)
    field = np.real(np.fft.ifftn(kernel * np.fft.fftn(chi)))
    affine = np.eye(4)
    affine[:3, :3] = _affine((1.0, 1.0, 1.0), 15.0, False)[:3, :3]
    affine[:3, 3] = -affine[:3, :3] @ np.array([32.0, 32.0, 32.0])   # world centre at voxel (32,32,32) of the world grid
    # sample the world-frame field on the tilted voxel grid, then bring it back
    idx = np.indices(shape).reshape(3, -1).astype(float)
    world = affine[:3, :3] @ idx + affine[:3, 3:4]
    native = ndimage.map_coordinates(field, world + 32.0, order=3, mode="nearest").reshape(shape).astype(np.float32)
    origin = np.array([-24.0, -24.0, -24.0])
    out = resample_to_world(native, affine, origin, (48, 48, 48), antialias=False)
    ref = field[8:56, 8:56, 8:56]
    assert np.linalg.norm((out - ref)[6:-6, 6:-6, 6:-6]) / np.linalg.norm(ref[6:-6, 6:-6, 6:-6]) < 0.02


def _write_subject(root, name, affine, shape, rng):
    chi = np.zeros(shape, np.float32)
    inner = tuple(slice(8, n - 8) for n in shape)
    chi[inner] = rng.normal(0, 0.03, size=chi[inner].shape).astype(np.float32)
    chi[inner] += 0.01
    for folder, data in (("qsm", chi), ("r2starmap", np.where(chi != 0, 0.03, 0.0)), ("t1w", np.where(chi != 0, 1000.0, 0.0))):
        (root / folder).mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(data.astype(np.float32), affine),
                 root / folder / f"{name}_{name}_ses-1_acq-wb_mod-{folder}_orient-std_brain.nii.gz")


def test_preprocess_subject_outputs(tmp_path):
    rng = np.random.default_rng(0)
    affine = _affine((0.7, 0.64, 0.64), 10.0, True)
    _write_subject(tmp_path, "sub-0000", affine, (50, 60, 60), rng)
    subjects = find_subjects(tmp_path)
    assert list(subjects) == ["sub-0000"]
    arrays, info = preprocess_subject(subjects["sub-0000"])
    mask = arrays["mask"]
    assert mask.shape == arrays["chi"].shape == arrays["weight"].shape and min(mask.shape) >= 96
    assert abs(arrays["chi"][mask].mean()) < 1e-6
    assert (arrays["chi"][~mask] == 0).all()
    assert arrays["weight"][mask].mean() == pytest.approx(0.264, abs=0.01)
    assert 0 <= arrays["weight"].min() and arrays["weight"].max() <= 1
    assert info["tilt_deg"] == pytest.approx(10.0, abs=0.01)


def test_preprocess_rejects_mismatched_affine(tmp_path):
    rng = np.random.default_rng(0)
    _write_subject(tmp_path, "sub-0001", _affine((0.7, 0.64, 0.64), 5.0, True), (40, 40, 40), rng)
    path = next((tmp_path / "t1w").glob("*.nii.gz"))
    image = nib.load(path)
    nib.save(nib.Nifti1Image(image.get_fdata(dtype=np.float32), _affine((0.7, 0.64, 0.64), 9.0, True)), path)
    with pytest.raises(ValueError, match="affine"):
        preprocess_subject(find_subjects(tmp_path)["sub-0001"])


def test_weight_rank_matching_reproduces_the_reference_distribution(tmp_path):
    from mcno_qsm.ahead import quantile_match

    rng = np.random.default_rng(1)
    values = rng.lognormal(0.0, 1.5, size=5000)
    reference = np.quantile(rng.beta(4.0, 11.0, size=20000), np.linspace(0, 1, 1001))
    matched = quantile_match(values, reference)
    assert np.allclose(np.percentile(matched, [1, 50, 99]), np.percentile(reference, [1, 50, 99]), atol=0.01)
    assert (np.argsort(values, kind="stable") == np.argsort(matched, kind="stable")).mean() > 0.99   # ranks kept (up to ties)
    _write_subject(tmp_path, "sub-0002", _affine((0.7, 0.64, 0.64), 10.0, True), (50, 60, 60), rng)
    arrays, _ = preprocess_subject(find_subjects(tmp_path)["sub-0002"], weight_reference=reference)
    w = arrays["weight"][arrays["mask"]]
    assert w.min() >= reference[0] - 1e-6 and w.max() <= reference[-1] + 1e-6
