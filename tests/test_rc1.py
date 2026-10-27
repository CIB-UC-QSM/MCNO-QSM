import builtins
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio
import torch
from scipy.ndimage import gaussian_filter

from mcno_qsm.artifacts import (CUBE_ROTATIONS, DEFAULT_WEIGHTS, FAMILIES, OBJECT_BOUND, ArtifactConfig, ObjectConfig,
                                Volume, add_shape, augment_object, dipole_field, make_dirty, phase_jumps,
                                rotation_matrix, warp)
from mcno_qsm.cleaner import PhaseCleaner
from mcno_qsm.rc1 import load_rc1, sti_fields
from mcno_qsm.rc1_data import RC1Source
from mcno_qsm.scoring import forward_field

N = 32
VOXEL = (1.0625, 1.0625, 1.0714)
DIPOLELET = dict(scales=2, scaling_law="static", base_angular_bands=2, delta=0.05, window="gaussian",
                 gaussian_width_factor=0.5, starlet="gaussian", starlet_sigma=0.5)


def write_fake_rc1(root: Path, n=N, seed=0):
    rng = np.random.default_rng(seed)
    smooth = lambda s: gaussian_filter(rng.standard_normal((n, n, n)), s).astype(np.float32)
    mask = np.zeros((n, n, n), np.uint8)
    mask[4:n - 4, 4:n - 4, 4:n - 4] = 1
    chi = smooth(2) * 0.05 * mask
    sio.savemat(root / "chi_cosmos.mat", {"chi_cosmos": chi})
    sio.savemat(root / "msk.mat", {"msk": mask})
    sio.savemat(root / "magn.mat", {"magn": (0.2 + 0.8 * np.abs(smooth(3)) / np.abs(smooth(3)).max()) * mask})
    sio.savemat(root / "phs_tissue.mat", {"phs_tissue": (chi * 0.3 + smooth(1) * 0.004).astype(np.float64)})
    sio.savemat(root / "spatial_res.mat", {"spatial_res": np.array([VOXEL])})
    sio.savemat(root / "chi_sti.mat", {"chi_13": smooth(2) * 0.01, "chi_23": smooth(2) * 0.01,
                                       "chi_33": chi * 1.1, "chi_sti": chi, "chi_sti_l2": chi})
    return root


@pytest.fixture(scope="module")
def fake_root(tmp_path_factory):
    return write_fake_rc1(tmp_path_factory.mktemp("rc1"))


@pytest.fixture(scope="module")
def volume(fake_root):
    return Volume.from_rc1(load_rc1(fake_root), "cpu")


def draw(family, volume, seed=0, **cfg):
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    return make_dirty(family, volume, rng, generator, ArtifactConfig(**cfg))


def test_rotation_matrix_is_orthonormal():
    r = rotation_matrix((37, 12, 71))
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12) and np.isclose(np.linalg.det(r), 1)


def test_ball_and_cylinder_volumes_and_axes():
    field = torch.zeros(41, 41, 41)
    add_shape(field, (20, 20, 20), 1.0, 6.0)
    assert abs(field.sum().item() - 4 / 3 * np.pi * 6**3) < 0.06 * 4 / 3 * np.pi * 6**3
    field = torch.zeros(41, 41, 41)
    add_shape(field, (20, 20, 20), 1.0, 5.0, 12.0)  # unrotated: axis along array axis 2 (B0)
    assert field.sum().item() == 13 * 81  # 13 voxel planes (|z| <= 6) of the 81-point disc of radius 5
    extent = [(field.sum(dim=tuple(d for d in range(3) if d != a)) > 0).sum().item() for a in range(3)]
    assert extent[0] == extent[1] and extent[2] == 13 and extent[0] == 11
    rotated = torch.zeros(41, 41, 41)
    add_shape(rotated, (20, 20, 20), 1.0, 5.0, 12.0, (0, 90, 0))  # axis moves to array axis 0
    extent = [(rotated.sum(dim=tuple(d for d in range(3) if d != a)) > 0).sum().item() for a in range(3)]
    assert extent[0] == 13 and extent[2] == 11


def test_shapes_clipped_at_the_border_do_not_fail():
    field = torch.zeros(10, 10, 10)
    add_shape(field, (0, 9, 5), 1.0, 4.0, 6.0, (20, 30, 40))
    add_shape(field, (50, 50, 50), 1.0, 4.0)  # entirely outside
    assert torch.isfinite(field).all() and field.sum() > 0


def test_target_is_masked_forward_of_chi_cosmos(fake_root, volume):
    data = load_rc1(fake_root)
    expected = forward_field(data["chi"], data["mask"].astype(np.float32), data["voxel_size"])
    assert np.allclose(volume.target.numpy(), expected, atol=1e-6)
    assert abs(data["chi"].mean()) < 1e-6  # global mean removed


def test_sti_fields_match_the_notebook_formula(fake_root):
    data = load_rc1(fake_root)
    res = np.array(VOXEL)
    n = data["chi_33"].shape
    # Notebook cell 5, written literally (even N: fftshift == ifftshift).
    ky, kx, kz = (np.arange(-m // 2, m // 2) for m in (n[1], n[0], n[2]))
    Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing="ij")
    Kx, Ky, Kz = (Kx / np.abs(Kx).max() / res[0], Ky / np.abs(Ky).max() / res[1], Kz / np.abs(Kz).max() / res[2])
    K2 = Kx**2 + Ky**2 + Kz**2
    K2[K2 == 0] = np.finfo(float).eps
    p13 = np.real(np.fft.ifftn(np.fft.fftshift(Kx * Kz / K2) * np.fft.fftn(data["chi_13"])))
    p23 = np.real(np.fft.ifftn(np.fft.fftshift(Ky * Kz / K2) * np.fft.fftn(data["chi_23"])))
    psti, pq = sti_fields(data["chi_13"], data["chi_23"], data["chi_33"], VOXEL)
    assert np.allclose(pq, p13 + p23, atol=1e-6)
    assert np.allclose(psti, forward_field(data["chi_33"], None, VOXEL) - (p13 + p23), atol=1e-6)


def test_dipole_field_matches_numpy_forward(volume):
    chi = volume.chi * 3
    assert np.allclose(dipole_field(volume, chi).numpy(), forward_field(chi.numpy(), None, VOXEL), atol=1e-5)


@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_is_finite_deterministic_and_seed_dependent(volume, family):
    a, info = draw(family, volume, seed=3)
    b, _ = draw(family, volume, seed=3)
    assert a.shape == volume.shape and torch.isfinite(a).all() and info["family"] == family
    assert torch.equal(a, b)
    if family not in ("tissue", "sti", "offdiag"):  # those three have no random part
        assert not torch.equal(a, draw(family, volume, seed=4)[0])


def test_phase_jumps_are_sparse_integers_inside_the_mask(volume):
    for seed in range(20):
        dirty, _ = draw("jumps", volume, seed=seed)
        jumps = dirty - volume.clean
        index = jumps.abs() > 1e-4
        assert 1 <= index.sum() <= 6
        assert set(jumps[index].abs().round().tolist()) <= {1.0, 2.0}
        assert volume.mask[index].min() == 1


def test_noise_family_shrinks_with_snr_and_leaves_outside_clean(volume):
    low, _ = draw("noise", volume, seed=1, snr_low=20, snr_high=20)
    high, _ = draw("noise", volume, seed=1, snr_low=200, snr_high=200)
    inside = volume.mask > 0
    err = lambda d: (d - volume.clean)[inside].square().mean().sqrt()
    assert err(high) < err(low) / 5
    assert torch.equal(low[~inside], volume.clean[~inside])
    zero, _ = draw("noise", volume, seed=1, outside="zero")
    assert zero[~inside].abs().max() == 0


def test_spheres_in_chi_add_a_dipole_field(volume):
    cfg = dict(snr_low=1e6, snr_high=1e6)  # negligible noise
    dirty, _ = draw("spheres_chi", volume, seed=5, **cfg)
    rng = np.random.default_rng(5)
    rng.uniform()  # sample_snr consumed one draw before the spheres
    from mcno_qsm.artifacts import spheres
    expected = volume.clean + dipole_field(volume, spheres(volume, rng, ArtifactConfig()))
    inside = volume.mask > 0
    assert torch.allclose(dirty[inside], expected[inside], atol=2e-5)


def test_patches_keep_field_and_target_consistent_through_augmentation(fake_root):
    source = RC1Source(fake_root, "cpu", ArtifactConfig(weights={"jumps": 1.0}), min_mask=0.0)
    rng, generator = np.random.default_rng(0), torch.Generator().manual_seed(0)
    for _ in range(10):
        field, mask, weight, target, info = source.patch(16, rng, generator)
        assert field.shape == mask.shape == weight.shape == target.shape == (1, 1, 16, 16, 16)
        psi = field * mask - target  # = mask * jumps: a few exact voxels, flips/rotations preserve that
        assert 0 <= (psi.abs() > 1e-4).sum() <= 6
        assert set(psi[psi.abs() > 1e-4].abs().round().tolist()) <= {1.0, 2.0}


def test_holdout_patches_stay_in_the_training_slab(fake_root):
    source = RC1Source(fake_root, "cpu", holdout={"axis": 2, "train_fraction": 0.6, "gap": 4}, min_mask=0.0)
    assert source.train_range == (0, 19) and source.test_range == (23, N)
    rng = np.random.default_rng(1)
    for _ in range(50):
        assert source.window(16, rng)[2] + 16 <= 19
    with pytest.raises(ValueError):
        source.window(24, rng)


def test_training_source_never_touches_rc2(fake_root, monkeypatch):
    opened = []
    real_open = builtins.open
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (opened.append(str(f)), real_open(f, *a, **k))[1])
    import nibabel
    monkeypatch.setattr(nibabel, "load", lambda f, *a, **k: opened.append(str(f)) or (_ for _ in ()).throw(
        AssertionError(f"unexpected nibabel.load({f})")))
    source = RC1Source(fake_root, "cpu", min_mask=0.0)
    rng, generator = np.random.default_rng(0), torch.Generator().manual_seed(0)
    for family in FAMILIES:
        source.patch(16, rng, generator, family)
    assert opened and not any("RC2" in p.upper() for p in opened)


def test_cleaner_takes_a_voxel_size_per_volume_and_plain_ablation_has_one_segment():
    torch.manual_seed(0)
    small = dict(channels=4, blocks=2, dilations=(1, 2), checkpoint=False)
    model = PhaseCleaner((1, 1, 1), dipolelet=DIPOLELET, **small)
    plain = PhaseCleaner((1, 1, 1), dipolelet=DIPOLELET, plain=True, **small)
    assert len(plain.blocks[0].real) == 1 and len(model.blocks[0].real) > 1
    assert sum(p.numel() for p in plain.parameters()) < sum(p.numel() for p in model.parameters())
    field = torch.randn(1, 1, 12, 12, 12)
    mask, weight = torch.ones_like(field), torch.ones_like(field)
    for m in (model, plain):
        assert torch.allclose(m(field, mask, weight, voxel_size=VOXEL), field * mask, atol=1e-6)  # identity at init
    with torch.no_grad():
        model.head.weight.normal_(0, 0.1)
    a = model(field, mask, weight)
    b = model(field, mask, weight, voxel_size=VOXEL)
    c = model(field, mask, weight, voxel_size=(1.0, 1.0, 1.0))
    assert torch.allclose(a, c) and not torch.allclose(a, b)  # default = built-in voxel size; geometry differs
    assert len(model._geometry_cache) == 2


def test_there_are_24_proper_cube_rotations():
    assert len(CUBE_ROTATIONS) == 24
    assert all(np.isclose(np.linalg.det(r), 1) and np.allclose(r @ r.T, np.eye(3)) for r in CUBE_ROTATIONS)


def array_rotations(x):
    """All 24 proper rotations of an array, by closure under rot90 (reflections are never produced)."""
    seen, frontier = {x.numpy().tobytes(): x}, [x]
    while frontier:
        current = frontier.pop()
        for dims in ((0, 1), (0, 2), (1, 2)):
            nxt = torch.rot90(current, 1, dims)
            if nxt.numpy().tobytes() not in seen:
                seen[nxt.numpy().tobytes()] = nxt
                frontier.append(nxt)
    return list(seen.values())


def test_warp_identity_and_cube_rotations_are_exact_proper_rotations(volume):
    chi = volume.chi
    assert torch.allclose(warp(chi, np.eye(3), 1.0), chi, atol=1e-6)
    reference = array_rotations(chi)
    assert len(reference) == 24
    matched = set()
    for rotation in CUBE_ROTATIONS:
        out = warp(chi, rotation, 1.0)
        hits = [i for i, t in enumerate(reference) if torch.allclose(out, t, atol=1e-6)]
        assert len(hits) == 1  # voxel centers map to voxel centers, and no mirror image appears
        matched.add(hits[0])
    assert len(matched) == 24


def test_augmented_object_is_a_consistent_head(volume):
    cfg = ObjectConfig(max_angle_deg=15.0)
    for seed in range(5):
        head = augment_object(volume, np.random.default_rng(seed), cfg)
        assert head is not None and head.shape == volume.shape
        assert set(head.mask.unique().tolist()) <= {0.0, 1.0}
        assert torch.equal(head.mask > 0, torch.from_numpy(head.mask_cpu))
        assert (head.chi[head.mask == 0] == 0).all()
        assert torch.allclose(head.target, head.mask * dipole_field(volume, head.chi), atol=1e-6)
    a = augment_object(volume, np.random.default_rng(7), cfg)
    assert torch.equal(a.chi, augment_object(volume, np.random.default_rng(7), cfg).chi)
    assert not torch.equal(a.chi, augment_object(volume, np.random.default_rng(8), cfg).chi)


def test_object_augmentation_gives_up_when_the_head_cannot_fit(volume):
    assert augment_object(volume, np.random.default_rng(0), ObjectConfig(scale=(3.0, 3.0), tries=3)) is None


def test_source_mixes_by_weight_and_keeps_real_psi_families_off_rotated_heads(fake_root):
    source = RC1Source(fake_root, "cpu", min_mask=0.0, objects=ObjectConfig(probability=1.0, max_angle_deg=5.0,
                                                                          scale=(1.0, 1.0)))
    assert sum(DEFAULT_WEIGHTS.values()) == 12 and DEFAULT_WEIGHTS["combo"] > DEFAULT_WEIGHTS["cylinders"]
    rng, generator = np.random.default_rng(0), torch.Generator().manual_seed(0)
    infos = [source.patch(16, rng, generator)[-1] for _ in range(60)]
    assert any(i["object_augmented"] for i in infos)
    assert not any(i["object_augmented"] and i["family"] in OBJECT_BOUND for i in infos)
    # An explicit real-psi family always uses the original head.
    assert not source.patch(16, rng, generator, "tissue")[-1]["object_augmented"]
    # Sampling frequencies follow the weights (original-pose source: all nine families).
    plain = RC1Source(fake_root, "cpu", min_mask=0.0, objects=ObjectConfig(probability=0.0))
    rng = np.random.default_rng(1)
    counts = {f: 0 for f in FAMILIES}
    for _ in range(3000):
        counts[plain._pick_family(rng, plain.families)] += 1
    for family, weight in DEFAULT_WEIGHTS.items():
        assert abs(counts[family] / 3000 - weight / 12) < 0.03


def test_holdout_rotated_heads_contain_no_held_out_anatomy(fake_root):
    source = RC1Source(fake_root, "cpu", holdout={"axis": 2, "train_fraction": 0.6, "gap": 4}, min_mask=0.0,
                       objects=ObjectConfig(probability=1.0))
    train_end = source.train_range[1]
    assert (source.train_volume.chi[:, :, train_end:] == 0).all() and (source.train_volume.mask[:, :, train_end:] == 0).all()
    assert source.volume.mask[:, :, train_end:].sum() > 0  # the full head, used for monitoring, keeps it
    # Rotated heads never grow beyond the truncated head's mass.
    head = augment_object(source.train_volume, np.random.default_rng(0), ObjectConfig(scale=(1.0, 1.0)))
    assert head.mask.sum() <= source.train_volume.mask.sum() * 1.01
