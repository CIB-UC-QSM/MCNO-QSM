"""Paper-recipe replication (CLAUDE.md 9bo): augmentation keys, voxel-size resampling, brain-free geometric training, fine-tuning, configs, tests."""

import importlib.util
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
import torch

from mcno_qsm.ahead_data import MixedSource
from mcno_qsm.artifacts import (CUBE_ROTATIONS, ArtifactConfig, GeometricConfig, ObjectConfig, Volume, augment_susceptibility,
                                dipole_field, geometric_chi, geometric_volume, make_dirty, resample_volume, sample_snr, scale_weight, warp)
from mcno_qsm.physics import dipole_kernel
from mcno_qsm.rc1 import load_rc1
from mcno_qsm.rc1_data import GeometricSource, RC1Source
from mcno_qsm.rc1_train import build_source, run
from test_ahead_data import ahead_config, write_fake_ahead
from test_rc1 import VOXEL, write_fake_rc1
from test_rc1_train import config, roots, rows  # noqa: F401  (roots is a fixture)

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name):
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(SCRIPTS))


@pytest.fixture(scope="module")
def volume(tmp_path_factory):
    root = tmp_path_factory.mktemp("rc1")
    write_fake_rc1(root)
    return Volume.from_rc1(load_rc1(root), "cpu")


def test_every_new_key_is_off_by_default_and_draws_nothing(volume):
    cfg = ArtifactConfig()
    rng = np.random.default_rng(0)
    state = rng.bit_generator.state
    assert augment_susceptibility(volume, rng, cfg) is volume and scale_weight(volume, rng, cfg) is volume
    assert rng.bit_generator.state == state and cfg.voxel_sizes is None and cfg.snr_mix is None
    a, b = np.random.default_rng(5), np.random.default_rng(5)
    assert sample_snr(a, cfg) == float(np.exp(b.uniform(np.log(cfg.snr_low), np.log(cfg.snr_high))))   # the original formula


def test_snr_mixture_follows_its_ranges_and_probabilities():
    cfg = ArtifactConfig(snr_mix=[[30.0, 50.0, 0.3], [90.0, 150.0, 0.7]])
    rng = np.random.default_rng(0)
    draws = np.array([sample_snr(rng, cfg) for _ in range(4000)])
    low, high = (draws >= 30) & (draws <= 50), (draws >= 90) & (draws <= 150)
    assert (low | high).all() and abs(low.mean() - 0.3) < 0.03 and abs(high.mean() - 0.7) < 0.03


def test_chi_scale_is_linear_and_recomputes_the_field(volume):
    out = augment_susceptibility(volume, np.random.default_rng(1), ArtifactConfig(chi_scale=(2.0, 2.0)))
    assert torch.allclose(out.chi, 2 * volume.chi) and torch.allclose(out.clean, 2 * volume.clean, atol=1e-5)
    assert torch.allclose(out.clean, dipole_field(volume, out.chi), atol=1e-6)
    assert torch.equal(out.magnitude, volume.magnitude) and torch.equal(out.mask, volume.mask)


def test_part_weights_scale_the_para_and_diamagnetic_parts_independently(volume):
    cfg = ArtifactConfig(chi_part_weight=(0.25, 4.0))
    out = augment_susceptibility(volume, np.random.default_rng(3), cfg)
    positive, negative = volume.chi > 0, volume.chi < 0
    p = (out.chi[positive] / volume.chi[positive])
    q = (out.chi[negative] / volume.chi[negative])
    assert torch.allclose(p, p[0].expand_as(p), rtol=1e-4) and torch.allclose(q, q[0].expand_as(q), rtol=1e-4)
    assert 0.25 <= p[0] <= 4.0 and 0.25 <= q[0] <= 4.0 and not torch.isclose(p[0], q[0])
    assert torch.allclose(out.clean, dipole_field(volume, out.chi), atol=1e-6)


def test_sign_inversion_flips_chi_and_field_with_the_given_probability(volume):
    always = augment_susceptibility(volume, np.random.default_rng(0), ArtifactConfig(chi_flip_prob=1.0))
    assert torch.allclose(always.chi, -volume.chi) and torch.allclose(always.clean, -volume.clean, atol=1e-5)
    rng = np.random.default_rng(0)
    flips = [float(augment_susceptibility(volume, rng, ArtifactConfig(chi_flip_prob=0.5)).chi.sum()) * float(volume.chi.sum()) < 0
             for _ in range(60)]
    assert 15 < sum(flips) < 45


def test_augmentation_is_seeded(volume):
    cfg = ArtifactConfig(chi_scale=(0.5, 2.0), chi_part_weight=(0.5, 2.0), chi_flip_prob=0.5)
    a = augment_susceptibility(volume, np.random.default_rng(9), cfg)
    assert torch.equal(a.chi, augment_susceptibility(volume, np.random.default_rng(9), cfg).chi)
    assert not torch.equal(a.chi, augment_susceptibility(volume, np.random.default_rng(10), cfg).chi)


def test_weight_scale_moves_the_weight_and_the_noise_follows_it(volume):
    inside = volume.mask > 0
    rms = {}
    for factor in (1.0, 3.0):
        v = scale_weight(volume, np.random.default_rng(0), ArtifactConfig(weight_scale=(factor, factor)))
        assert torch.allclose(v.magnitude, factor * volume.magnitude)
        dirty, _ = make_dirty("noise", v, np.random.default_rng(2), torch.Generator().manual_seed(2), ArtifactConfig(snr_low=15, snr_high=15))
        rms[factor] = (dirty - v.clean)[inside].square().mean().sqrt().item()
    assert 2.4 < rms[1.0] / rms[3.0] < 3.6    # phase noise ~ 1 / (snr * magnitude)


def test_resampled_head_is_consistent(volume):
    for voxel in (0.8, 1.0625, 2.0):
        v = resample_volume(volume, voxel, min_side=24)
        side = v.shape[0]
        assert v.shape == (side,) * 3 and side >= 24 and v.voxel_size == (voxel,) * 3
        assert set(v.mask.unique().tolist()) <= {0.0, 1.0} and torch.equal(v.mask > 0, torch.from_numpy(v.mask_cpu))
        assert (v.chi[v.mask == 0] == 0).all() and (v.magnitude[v.mask == 0] == 0).all()
        assert torch.allclose(v.clean, dipole_field(v, v.chi), atol=1e-6)
        assert v.kernel.shape == v.shape and torch.allclose(v.kernel, dipole_kernel(v.shape, (voxel,) * 3))
    # same physical head: the brain volume (mm^3) is kept within a few percent when coarsening, and the mean weight is not diluted
    coarse = resample_volume(volume, 2.0)
    assert abs(coarse.mask.sum() * 2.0**3 / (volume.mask.sum() * 1.0625**3) - 1) < 0.1
    mean = lambda v: float(v.magnitude[v.mask > 0].mean())
    assert abs(mean(coarse) / mean(volume) - 1) < 0.1


def test_resampling_to_the_native_size_keeps_the_chi(volume):
    same = resample_volume(volume, float(np.mean(volume.voxel_size)))
    assert same.shape == volume.shape and torch.equal(same.mask, volume.mask)
    assert torch.allclose(same.chi, volume.chi * volume.mask, atol=1e-5)       # chi outside the mask is dropped (as in augment_object)


def test_the_z_axis_of_the_object_can_move_to_x_and_y():
    # B0 stays on array axis 2; a rod along axis 2 is turned onto axis 0 or 1 by some of the 24 cube rotations (permutation of the axes)
    rod = torch.zeros(21, 21, 21)
    rod[10, 10, 3:18] = 1.0
    extent = lambda t: [int((t.sum(dim=tuple(d for d in range(3) if d != a)) > 0.5).sum()) for a in range(3)]
    assert extent(rod) == [1, 1, 15]
    seen = set()
    for rotation in CUBE_ROTATIONS:
        e = extent(warp(rod, rotation, 1.0))
        seen.add(int(np.argmax(e)))
    assert seen == {0, 1, 2}


def make_source(root, **cfg):
    weights = cfg.pop("weights", {"noise": 1.0, "clean": 0.25})
    return RC1Source(root, "cpu", ArtifactConfig(weights=weights, **cfg), min_mask=0.0, objects=ObjectConfig(probability=1.0, max_angle_deg=5.0))


def test_a_batch_shares_one_voxel_size_and_the_default_batch_keeps_the_native_one(tmp_path):
    root = tmp_path / "rc1"
    root.mkdir()
    write_fake_rc1(root)
    rng, generator = np.random.default_rng(0), torch.Generator().manual_seed(0)
    native = make_source(root)
    assert native.next_batch(16, rng, generator, 2)[-1] == VOXEL
    source = make_source(root, voxel_sizes=(1.0, 2.0, 3.0), chi_scale=(0.5, 2.0))
    voxels = set()
    side = {(1.0,) * 3: 16, (2.0,) * 3: 16, (3.0,) * 3: 8}      # the 32^3 head of 1.06 mm is 17 voxels at 2 mm and 11 at 3 mm: the patch shrinks to fit
    for _ in range(12):
        field, mask, weight, target, voxel = source.next_batch(16, rng, generator, 2)
        assert voxel in side and torch.isfinite(field).all()
        assert field.shape == mask.shape == weight.shape == target.shape == (2, 1, *(side[voxel],) * 3)
        voxels.add(voxel)
    assert len(voxels) >= 2
    with pytest.raises(ValueError):
        make_source(root, voxel_sizes=(1.0,), weights={"tissue": 1.0})


def test_recipe_keys_change_the_patch_and_the_weight_channel(tmp_path):
    root = tmp_path / "rc1"
    root.mkdir()
    write_fake_rc1(root)
    plain = make_source(root, snr_low=1e6, snr_high=1e6)
    scaled = make_source(root, snr_low=1e6, snr_high=1e6, weight_scale=(2.0, 2.0), chi_scale=(2.0, 2.0))
    a = plain.patch(16, np.random.default_rng(4), torch.Generator().manual_seed(4), "clean")
    b = scaled.patch(16, np.random.default_rng(4), torch.Generator().manual_seed(4), "clean")
    # the recipe consumes rng draws, so the placement differs: compare means. Weight channel x2 (the mean is over the mask of the patch)
    assert abs(b[2][b[1] > 0].mean() / a[2][a[1] > 0].mean() - 2) < 0.5


def test_real_psi_fields_are_never_rescaled(tmp_path):
    root = tmp_path / "rc1"
    root.mkdir()
    write_fake_rc1(root)
    source = make_source(root, chi_scale=(3.0, 3.0), chi_flip_prob=1.0, weights={"tissue": 1.0})
    source.objects = ObjectConfig(probability=0.0)
    field, mask, weight, target, info = source.patch(16, np.random.default_rng(0), torch.Generator().manual_seed(0), "tissue")
    unscaled = make_source(root, weights={"tissue": 1.0})
    unscaled.objects = ObjectConfig(probability=0.0)
    f2, _, _, _, _ = unscaled.patch(16, np.random.default_rng(0), torch.Generator().manual_seed(0), "tissue")
    assert torch.equal(field, f2) and torch.equal(weight, unscaled.patch(16, np.random.default_rng(0), torch.Generator().manual_seed(0), "tissue")[2])


# ---------------------------------------------------------------- geometric figures (E2)

def test_geometric_volume_is_a_consistent_brain_free_case():
    cfg = GeometricConfig()
    kernel = dipole_kernel((32,) * 3, VOXEL)
    v = geometric_volume(32, VOXEL, np.random.default_rng(0), cfg, 19.0, kernel, "cpu")
    assert v.shape == (32,) * 3 and torch.isfinite(v.clean).all() and v.chi.abs().max() > 0
    assert 0.15 < float(v.mask.mean()) < 0.6                                   # ellipsoidal mask, a head-sized share of the cube
    assert torch.allclose(v.clean, dipole_field(v, v.chi), atol=1e-6)
    assert torch.equal(v.mask > 0, torch.from_numpy(v.mask_cpu)) and (v.magnitude[v.mask == 0] == 0).all()
    mean_weight = float(v.magnitude[v.mask > 0].mean())
    assert 0.1 < mean_weight < 0.6
    again = geometric_volume(32, VOXEL, np.random.default_rng(0), cfg, 19.0, kernel, "cpu")
    assert torch.equal(v.chi, again.chi) and torch.equal(v.magnitude, again.magnitude)
    assert not torch.equal(v.chi, geometric_volume(32, VOXEL, np.random.default_rng(1), cfg, 19.0, kernel, "cpu").chi)
    ones = geometric_volume(32, VOXEL, np.random.default_rng(0), GeometricConfig(mask="ones", magnitude_variation=0.0), 19.0, kernel, "cpu")
    assert ones.mask.min() == 1 and torch.allclose(ones.magnitude, torch.full_like(ones.magnitude, 0.264))


def test_geometric_figures_are_spheres_and_cylinders_with_both_signs():
    rng = np.random.default_rng(0)
    mask = np.ones((32,) * 3, bool)
    signs = set()
    for _ in range(20):
        chi = geometric_chi(32, rng, GeometricConfig(), mask, "cpu")
        values = chi[chi != 0]
        assert values.numel() > 0 and values.abs().max() <= 0.4 * 20          # additive overlaps are bounded by the figure count
        signs |= {float(values.min()) < 0, float(values.max()) > 0}
    assert signs == {True}


def test_geometric_source_uses_no_rc1_voxel_in_training_patches(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    write_fake_rc1(a, seed=0)
    write_fake_rc1(b, seed=1)                                                   # a different head, same TE / B0 / voxel size
    make = lambda root: GeometricSource(root, "cpu", ArtifactConfig(weights={"noise": 1.0, "clean": 0.25}), min_mask=0.0)
    sa, sb = make(a), make(b)
    ra, rb = np.random.default_rng(3), np.random.default_rng(3)
    ga, gb = torch.Generator().manual_seed(3), torch.Generator().manual_seed(3)
    for _ in range(4):
        pa, pb = sa.next_batch(24, ra, ga, 2), sb.next_batch(24, rb, gb, 2)
        assert pa[0].shape == (2, 1, 24, 24, 24) and pa[-1] == VOXEL
        assert all(torch.equal(x, y) for x, y in zip(pa[:4], pb[:4]))
    assert sa.scale != sb.scale                                                 # the monitoring side does read the head
    with pytest.raises(ValueError):
        sa.patch(24, ra, ga, "tissue")
    with pytest.raises(ValueError):
        GeometricSource(a, "cpu", ArtifactConfig(weights={"tissue": 1.0}))


def test_geometric_source_takes_the_recipe_keys(tmp_path):
    root = tmp_path / "rc1"
    root.mkdir()
    write_fake_rc1(root)
    source = GeometricSource(root, "cpu", ArtifactConfig(weights={"noise": 1.0}, voxel_sizes=(1.5, 2.5), weight_scale=(0.5, 3.0),
                                                         chi_part_weight=(0.5, 2.0), snr_mix=[[30, 50, 0.3], [90, 150, 0.7]]), min_mask=0.0)
    rng, generator = np.random.default_rng(0), torch.Generator().manual_seed(0)
    voxels = {source.next_batch(24, rng, generator, 2)[-1] for _ in range(10)}
    assert voxels <= {(1.5,) * 3, (2.5,) * 3} and len(voxels) == 2


def geo_config(rc1, rc2, output, steps):
    cfg = config(rc1, rc2, output, steps, weights={"noise": 1.0, "clean": 0.25})
    cfg["data"]["geometric"] = {}
    return cfg


def test_geometric_run_trains_and_is_selected_on_the_rc1_check_slab(roots, tmp_path):
    rc1, rc2 = roots
    summary = run(geo_config(rc1, rc2, tmp_path / "geo", 4))
    table = rows(tmp_path / "geo" / "eval.csv")
    assert {"noise_test", "clean_test", "sel_trained", "sim2_nrmse"} <= set(table[0]) and summary["steps"] == 4
    assert isinstance(build_source(geo_config(rc1, rc2, tmp_path / "x", 1), "cpu"), GeometricSource)


# ---------------------------------------------------------------- AHEAD subjects and fine-tuning (E1 on one head, E3)

def test_ahead_n_train_keeps_a_seeded_subset_and_voxel_sizes_apply(roots, tmp_path):
    rc1, rc2 = roots
    ahead = write_fake_ahead(tmp_path / "ahead", n=8)
    cfg = ahead_config(rc1, rc2, ahead, tmp_path / "o", 1, fraction=1.0)
    cfg["data"]["ahead"]["n_train"] = 2
    cfg["artifacts"]["voxel_sizes"] = [1.5, 2.0]
    cfg["artifacts"]["chi_scale"] = [0.5, 2.0]
    source = build_source(cfg, "cpu")
    assert isinstance(source, MixedSource) and len(source.train_names) == 2 and set(source.raw) == set(source.train_names)
    again = build_source(cfg, "cpu")
    assert again.train_names == source.train_names
    rng, generator = np.random.default_rng(0), torch.Generator().manual_seed(0)
    voxels = set()
    for _ in range(8):
        field, mask, weight, target, voxel = source.next_batch(16, rng, generator, 2)
        assert field.shape == (2, 1, 16, 16, 16) and torch.isfinite(field).all()
        voxels.add(voxel)
    assert voxels <= {(1.5,) * 3, (2.0,) * 3}
    cfg["data"]["ahead"].pop("n_train")
    assert len(build_source(cfg, "cpu").train_names) == 6


def test_init_from_starts_from_the_given_weights_and_resume_ignores_it(roots, tmp_path):
    rc1, rc2 = roots
    run(config(rc1, rc2, tmp_path / "base", 3))
    base = torch.load(tmp_path / "base" / "best.pt", weights_only=False)["model"]
    cfg = config(rc1, rc2, tmp_path / "ft", 3)
    cfg["training"].update(learning_rate=0.0, init_from=str(tmp_path / "base" / "best.pt"))
    run(cfg)
    ft = torch.load(tmp_path / "ft" / "final.pt", weights_only=False)["model"]
    assert all(torch.equal(base[k], ft[k]) for k in base)                      # lr 0: nothing moved away from the initial weights
    assert "init_from" in (tmp_path / "ft" / "metadata.json").read_text()
    plain = config(rc1, rc2, tmp_path / "plain", 3)
    plain["training"]["learning_rate"] = 0.0
    run(plain)
    other = torch.load(tmp_path / "plain" / "final.pt", weights_only=False)["model"]
    assert any(not torch.equal(base[k], other[k]) for k in base)               # without init_from the (seeded) start is the fresh init


def test_l2_only_loss_runs(roots, tmp_path):
    rc1, rc2 = roots
    cfg = config(rc1, rc2, tmp_path / "l2", 3)
    cfg["training"]["l1_weight"] = 0.0
    run(cfg)
    history = rows(tmp_path / "l2" / "history.csv")
    assert all(abs(float(r["loss"]) - float(r["l2"])) < 1e-6 for r in history)


# ---------------------------------------------------------------- configs and the paper tests

def test_generated_paper_configs_parse_and_use_known_keys():
    module = load_script("make_paper_configs")
    assert set(module.RUNS) == {"paper_e1_rc1", "paper_e1_ahead1", "paper_e2_geo", "paper_e3_ft", "paper_e4_l2only", "paper_ws", "paper_e1_rc1_ws"}
    for name, spec in module.RUNS.items():
        cfg = tomllib.loads(module.render(name, spec))
        art = cfg["artifacts"]
        ArtifactConfig(**{k: v for k, v in art.items() if k != "weights"})
        ObjectConfig(**cfg["objects"])
        GeometricConfig(**cfg["data"].get("geometric", {}))
        assert cfg["training"]["output"] == f"runs/s4b_{name}" and cfg["training"]["seed"] == 42
        assert set(art["weights"]) == {"noise", "clean"}
    e2 = tomllib.loads(module.render("paper_e2_geo", module.RUNS["paper_e2_geo"]))
    assert e2["training"]["steps"] * e2["training"]["batch_size"] == 30000 and e2["training"]["patch"] == 64 and "geometric" in e2["data"]
    assert (e2["artifacts"]["snr_low"], e2["artifacts"]["snr_high"]) == (30.0, 120.0)
    e3 = tomllib.loads(module.render("paper_e3_ft", module.RUNS["paper_e3_ft"]))
    assert e3["training"]["init_from"].endswith("s4b_paper_e2_geo/best.pt") and e3["data"]["ahead"]["n_train"] == 2
    assert e3["artifacts"]["snr_mix"] == [[30.0, 50.0, 0.3], [90.0, 150.0, 0.7]] and e3["artifacts"]["chi_flip_prob"] == 0.5
    e1 = tomllib.loads(module.render("paper_e1_ahead1", module.RUNS["paper_e1_ahead1"]))
    assert e1["data"]["ahead"]["n_train"] == 1 and e1["data"]["ahead"]["fraction"] == 1.0
    assert tomllib.loads(module.render("paper_e4_l2only", module.RUNS["paper_e4_l2only"]))["training"]["l1_weight"] == 0.0
    # the same keys as the reference noise_clean recipe everywhere else
    reference = tomllib.loads((Path(__file__).resolve().parents[1] / "configs" / "s4b_noise_clean.toml").read_text())
    ws = tomllib.loads(module.render("paper_ws", module.RUNS["paper_ws"]))
    for table in ("dipolelet", "model", "objects", "data"):
        assert ws[table] == reference[table]
    assert {k: v for k, v in ws["training"].items() if k != "output"} == {k: v for k, v in reference["training"].items() if k != "output"}


@pytest.fixture(scope="module")
def paper_data(tmp_path_factory):
    root = tmp_path_factory.mktemp("rc1_paper")
    write_fake_rc1(root)
    return load_rc1(root)


def test_paper_tests_fields(paper_data):
    module = load_script("export_paper_tests")
    fields = module.paper_fields(paper_data, "mag_base", psi_rms=0.25, psi_sigma=4.0)  # fake chi is weak: a large psi is needed to wrap
    assert list(fields) == ["paper_snr40", "paper_snr100", "paper_snr100_wrap"]
    mask = paper_data["mask"]
    s = paper_data["phs_scale"]
    from mcno_qsm.scoring import forward_field
    clean = forward_field(paper_data["chi"], None, paper_data["voxel_size"])
    err = {k: float(np.sqrt(np.mean((v[0] - clean)[mask] ** 2))) for k, v in fields.items()}
    assert err["paper_snr40"] > err["paper_snr100"]                             # more noise at SNR 40
    assert fields["paper_snr100_wrap"][2]["wrapped_fraction"] > 0.2             # the strong inconsistency wraps a large share of the brain
    assert fields["paper_snr100"][2]["wrapped_fraction"] == 0.0                 # COSMOS alone never wraps (|b| < pi / s)
    assert np.abs(fields["paper_snr100_wrap"][0][mask]).max() <= np.pi / s + 1e-6   # a wrapped phase stays inside +-pi
    assert np.array_equal(fields["paper_snr40"][0], module.paper_fields(paper_data, "mag_base", psi_sigma=4.0)["paper_snr40"][0])
    base = module.paper_fields(paper_data, "magn")["paper_snr100"][1]
    assert np.allclose(base, paper_data["magnitude"])


def test_wrap_difference_ignores_multiples_of_two_pi():
    module = load_script("export_paper_tests")
    s = 19.0
    target = np.linspace(-0.2, 0.2, 50)
    shifted = target + 2 * np.pi / s * np.arange(50) % 3
    assert np.allclose(module.wrap_difference(target + 2 * np.pi / s * 2, target, s), 0, atol=1e-9)
    assert np.allclose(module.wrap_difference(target + 0.01, target, s), 0.01, atol=1e-9)
    assert np.abs(module.wrap_difference(shifted, target, s)).max() < np.pi / s + 1e-9


def test_fit_size_shrinks_the_patch_to_the_resampled_head():
    fit = RC1Source.fit_size
    assert fit((160,) * 3, (1.0625,) * 3, 1.0, 96) == 96 and fit((160,) * 3, (1.0625,) * 3, 2.0, 96) == 80
    assert fit((160,) * 3, (1.0625,) * 3, 3.0, 96) == 56 and fit((10,) * 3, (1.0,) * 3, 3.0, 96) == 8
    assert fit((160,) * 3, (1.0625,) * 3, 1.0625, 64) == 64            # never above the requested patch


def test_training_with_voxel_sizes_uses_the_actual_patch_size_in_every_cache(roots, tmp_path):
    rc1, rc2 = roots
    cfg = config(rc1, rc2, tmp_path / "vox", 4)
    cfg["artifacts"].update(voxel_sizes=[1.0, 3.0], chi_scale=[0.5, 2.0], weights={"noise": 1.0, "clean": 0.25})
    cfg["training"].update(band_weight=1.0, cone_weight=0.1)
    run(cfg)
    assert all(float(r["band"]) > 0 for r in rows(tmp_path / "vox" / "history.csv"))
