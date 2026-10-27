import numpy as np
import torch

from mcno_qsm.intravoxel import block_mean, dephase, fine_field, source_chi_fine

PER_PPM_S = 2 * np.pi * 42.58 * 7.0


def test_block_mean_of_constant_and_shape():
    x = torch.full((8, 8, 8), 3.0, dtype=torch.float64)
    out = block_mean(x, 4)
    assert out.shape == (2, 2, 2) and torch.allclose(out, torch.full_like(out, 3.0))


def test_no_source_keeps_background_phase_and_magnitude():
    bg = torch.full((6, 6, 6), 0.02, dtype=torch.float64)
    mag = torch.full_like(bg, 0.4)
    zero = torch.zeros(24, 24, 24, dtype=torch.float64)
    signal, target = dephase(bg, mag, zero, 4, 1.0, 0.02, PER_PPM_S)
    assert torch.allclose(target, bg, atol=1e-12)
    assert torch.allclose(signal.abs(), mag, atol=1e-12)
    assert torch.allclose(torch.angle(signal), bg * 0.02 * PER_PPM_S, atol=1e-9)


def test_strong_source_loses_magnitude_and_more_at_later_echoes():
    shape = (16, 16, 16)
    bg = torch.zeros(shape, dtype=torch.float64)
    mag = torch.full(shape, 0.4, dtype=torch.float64)
    src = source_chi_fine(shape, 4, (8.0, 8.0, 8.0), (1.5, 1.5, 1.5), -1.5)
    ratios = []
    for te in (0.004, 0.028):
        signal, _ = dephase(bg, mag, src, 4, 1.0, te, PER_PPM_S)
        ratios.append(float(signal.abs()[8, 8, 6:11].min() / 0.4))
    assert ratios[0] < 0.95 and ratios[1] < ratios[0]


def test_source_field_is_odd_around_equator_and_dipolar():
    src = source_chi_fine((12, 12, 12), 2, (6.0, 6.0, 6.0), (1.5, 1.5, 1.5), -1.0)
    field = fine_field(src, 0.5)
    # a sphere of negative chi has a field with the dipole sign pattern: opposite signs along z and in the transverse plane
    assert field[12, 12, 20] * field[20, 12, 12] < 0


# ---- families `acq` and `intravoxel`, noise statistics (CLAUDE.md 9bp) ----
import csv

from mcno_qsm.acquisition import complex_noise, estimate_ppm, per_ppm_per_s, weight_prime
from mcno_qsm.artifacts import ACQ, EXTRA_FAMILIES, FAMILIES, INTRAVOXEL, ArtifactConfig, apply_overrides, make_dirty
from mcno_qsm.noise import simulate_dirty
from mcno_qsm.rc1_train import run
from test_rc1 import fake_root, volume  # noqa: F401  (fixtures)
from test_rc1_train import config, roots  # noqa: F401

SMALL = dict(iv_box=16, iv_factor=2)


def sample(family, volume, seed=0, **cfg):  # noqa: F811
    rng, generator = np.random.default_rng(seed), torch.Generator().manual_seed(seed)
    return make_dirty(family, volume, rng, generator, ArtifactConfig(**cfg))


def test_new_families_are_extra_not_monitored_nine():
    assert ACQ in EXTRA_FAMILIES and INTRAVOXEL in EXTRA_FAMILIES and ACQ not in FAMILIES and INTRAVOXEL not in FAMILIES


def test_echo_fit_recovers_a_noise_free_field_and_aliases_beyond_the_unwrapping_limit():
    tes, b0 = (0.004, 0.012, 0.020, 0.028), 7.0
    field = torch.tensor([0.0, 0.05, -0.1, 0.19, 0.25, -0.3], dtype=torch.float64).reshape(-1, 1, 1)
    t = torch.tensor(tes, dtype=torch.float64).reshape(-1, 1, 1, 1)
    signal = torch.exp(1j * field[None] * t * per_ppm_per_s(b0))
    est = estimate_ppm(signal, tes, b0)
    assert torch.allclose(est[:4], field[:4], atol=1e-9)           # below pi / (2 pi gamma B0 dTE) = 0.21 ppm
    assert (est[4:] - field[4:]).abs().min() > 0.05                # beyond it the temporal unwrap fails (Sim2's saturation at the source)


def test_weight_prime_matches_the_sim2_definition():
    m = torch.tensor([0.4, 0.3, 0.2, 0.1]).reshape(4, 1, 1, 1)
    te = (0.004, 0.012, 0.020, 0.028)
    expected = sum(float(x) ** 2 * t for x, t in zip(m.flatten(), te)) / sum(float(x) * t for x, t in zip(m.flatten(), te))
    assert abs(float(weight_prime(m, te)) - expected) < 1e-6


def test_noise_kinds_have_unit_variance_and_their_own_structure():
    shape = (1, 40, 40, 40)
    g = lambda: torch.Generator().manual_seed(0)
    plain = complex_noise(shape, None, g(), "cpu")
    assert abs(float(plain.real.std()) - 1) < 0.02 and abs(float(plain.imag.std()) - 1) < 0.02
    colored = complex_noise(shape, {"kind": "colored", "sigma": 1.2}, g(), "cpu")
    assert abs(float(colored.real.std()) - 1) < 0.05
    lag = float((colored.real[0, 1:] * colored.real[0, :-1]).mean() / colored.real.var())
    assert lag > 0.4 and abs(float((plain.real[0, 1:] * plain.real[0, :-1]).mean())) < 0.03
    heavy = complex_noise(shape, {"kind": "heavy", "df": 3}, g(), "cpu")
    kurtosis = float(((heavy.real - heavy.real.mean()) ** 4).mean() / heavy.real.var() ** 2)
    assert abs(float(heavy.real.std()) - 1) < 0.3 and kurtosis > 4.5
    hetero = complex_noise(shape, {"kind": "hetero", "depth": 0.7}, g(), "cpu")
    blocks = hetero.real[0].reshape(2, 20, 2, 20, 2, 20).std(dim=(1, 3, 5))
    assert float(blocks.max() / blocks.min()) > 1.3 and abs(float(hetero.real.square().mean()) - 1) < 0.1


def test_default_noise_family_is_unchanged_by_the_new_keys(volume):  # noqa: F811
    dirty, info = sample("noise", volume, snr_low=50.0, snr_high=50.0)
    assert "noise" not in info
    generator = torch.Generator().manual_seed(0)
    expected = simulate_dirty(volume.clean, volume.magnitude * volume.mask, 50.0, volume.phs_scale, generator)
    assert torch.equal(dirty, torch.where(volume.mask > 0, expected, volume.clean))


def test_noise_family_accepts_other_statistics(volume):  # noqa: F811
    dirty, info = sample("noise", volume, noise_kinds=(("heavy", 1.0),))
    assert info["noise"] == "heavy" and torch.isfinite(dirty).all()


def test_acq_without_noise_returns_the_clean_field(volume):  # noqa: F811
    dirty, info = sample(ACQ, volume, snr_low=1e9, snr_high=1e9)
    assert info["sources"] == 0 and torch.equal(info["clean"], volume.clean)
    inside = volume.mask > 0
    assert float((dirty - volume.clean)[inside].abs().max()) < 1e-3
    assert torch.equal(dirty[~inside], volume.clean[~inside])               # outside the mask the field stays noise-free


def test_intravoxel_adds_a_source_whose_field_is_missing_from_the_input_and_dips_the_weight(volume):  # noqa: F811
    dirty, info = sample(INTRAVOXEL, volume, snr_low=1e9, snr_high=1e9, iv_count=(1, 1), iv_ppm=(2.0, 2.0), iv_radius=(2.0, 2.0),
                         iv_positive_prob=0.0, **SMALL)
    assert info["sources"] == 1
    added = (info["clean"] - volume.clean) * volume.mask
    assert float(added.abs().max()) > 0.02                                   # the target contains the source field
    assert torch.equal(apply_overrides(volume, info).target, volume.mask * info["clean"])
    ratio = info["magnitude"][volume.mask > 0] / volume.magnitude[volume.mask > 0]
    assert float(ratio.min()) < 0.8 and float(ratio.max()) <= 1.0 + 1e-6    # the weight dips only
    gap = (dirty - info["clean"]) * volume.mask
    assert float(gap.abs().max()) > 0.1                                      # near the source the input differs strongly from the target
    far = added.abs() < 1e-4
    assert float(gap[far & (volume.mask > 0)].abs().max()) < 0.05            # far from it the noise-free input follows the target


def test_intravoxel_is_deterministic_in_its_seeds(volume):  # noqa: F811
    a, _ = sample(INTRAVOXEL, volume, seed=3, **SMALL)
    b, _ = sample(INTRAVOXEL, volume, seed=3, **SMALL)
    c, _ = sample(INTRAVOXEL, volume, seed=4, **SMALL)
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_trainer_runs_with_the_new_families_and_logs_them(roots, tmp_path):  # noqa: F811
    rc1, rc2 = roots
    cfg = config(rc1, rc2, tmp_path / "run", 3, weights={"noise": 1.0, "clean": 0.25, "acq": 0.5, "intravoxel": 0.5})
    cfg["artifacts"].update(SMALL, noise_kinds=[["gaussian", 0.5], ["heavy", 0.5]])
    run(cfg)
    with open(tmp_path / "run" / "eval.csv", newline="") as stream:
        row = list(csv.DictReader(stream))[-1]
    assert {"acq_test", "intravoxel_test", "intravoxel_in_test"} <= set(row)
    assert np.isfinite(float(row["intravoxel_test"])) and float(row["intravoxel_in_test"]) > 0
