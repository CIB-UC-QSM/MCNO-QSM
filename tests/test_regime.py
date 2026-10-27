"""Coarse split into |D| bands, regime-gated linear Dipole-let filter and the noisy real-residual families (CLAUDE.md 9ar)."""
import pytest
import torch

from mcno_qsm.artifacts import OBJECT_BOUND, REAL_NOISE
from mcno_qsm.cleaner import PhaseCleaner, RegimeFilter
from mcno_qsm.dipolelets import SIM2_DIPOLELET, DipoleletTransform
from mcno_qsm.rc1_train import run
from test_cleaner import DIPOLELET, SMALL, inputs
from test_rc1 import draw, fake_root, volume  # noqa: F401  (fixtures)
from test_rc1_train import config, rows, roots  # noqa: F401  (roots is a fixture)

SPLIT = {**DIPOLELET, "coarse_bands": 3}
SHAPE = (24, 24, 24)


def test_default_transform_is_unchanged_and_the_split_adds_two_segments():
    base = DipoleletTransform(SHAPE, (1, 1, 1), **DIPOLELET)
    split = DipoleletTransform(SHAPE, (1, 1, 1), **SPLIT)
    assert base.segment_names[-1] == "coarse" and len(base.segment_names) == 5
    assert split.segment_names[-3:] == ["coarse/band0", "coarse/band1", "coarse/band2"] and len(split.segment_names) == 7
    x = torch.randn(1, 1, *SHAPE)
    assert torch.allclose(DipoleletTransform.synthesis(split.analysis(x)), x, atol=1e-5)   # exact reconstruction (partition of unity)
    # the detail segments are identical, only the coarse one is divided
    for a, b in zip(list(base.analysis(x))[:-1], list(split.analysis(x))[:-1]):
        assert torch.equal(a, b)
    assert torch.allclose(sum(list(split.analysis(x))[-3:]), list(base.analysis(x))[-1], atol=1e-5)


def test_sim2_parameters_with_the_split_give_twelve_segments():
    tr = DipoleletTransform((16, 16, 16), (1, 1, 1), **{**SIM2_DIPOLELET, "coarse_bands": 3})
    assert len(tr.segment_names) == 12


def make(regime=None, dipolelet=SPLIT, **kw):
    return PhaseCleaner((1, 1, 1), dipolelet=dipolelet, regime=regime, **{**SMALL, **kw})


def test_regime_filter_is_the_identity_at_init_and_adds_one_input_channel():
    torch.manual_seed(0)
    model = make(dict(rho=2.0))
    field, mask, weight = inputs(SHAPE)
    assert torch.allclose(model(field, mask, weight), field * mask, atol=1e-6)
    assert model.lift[0].in_channels == 4 and make().lift[0].in_channels == 3
    assert "regime.g0" in model.state_dict() and not any(k.startswith("regime") for k in make().state_dict())


def test_regime_filter_removal_follows_the_gains_exactly():
    field, mask, weight = inputs(SHAPE)
    x = field * mask
    # g0 = g1 = 0 removes every segment: the whole input
    zero = make(dict(rho=2.0, g0=[0.0] * 7, g1=[0.0] * 7))
    assert torch.allclose(zero(field, mask, weight), torch.zeros_like(x), atol=1e-5)
    # a gain of 0.5 on the whole spectrum halves the input (the segments sum to the input)
    half = make(dict(rho=2.0, g0=[0.5] * 7, g1=[0.5] * 7))
    assert torch.allclose(half(field, mask, weight), 0.5 * x, atol=1e-5)
    # r = 1 (large c) selects g1, r = 0 (very negative c) selects g0
    g0, g1 = [1.0] * 7, [0.0] * 7
    on = make(dict(rho=2.0, g0=g0, g1=g1, c=50.0))
    off = make(dict(rho=2.0, g0=g0, g1=g1, c=-50.0))
    assert torch.allclose(on(field, mask, weight), torch.zeros_like(x), atol=1e-4)
    assert torch.allclose(off(field, mask, weight), x, atol=1e-4)


def cone_heavy(shape, strength):
    """White noise plus a field concentrated in the cone (|D| < 0.05) of every scale."""
    tr = DipoleletTransform(shape, (1, 1, 1), **SPLIT)
    g = torch.Generator().manual_seed(5)
    white = torch.randn(1, 1, *shape, generator=g)
    parts = list(tr.analysis(white))
    cone = sum(p for name, p in zip(tr.segment_names, parts) if name.endswith("band0"))
    return white + strength * torch.fft.ifftn(cone, dim=(-3, -2, -1), norm="ortho").real


def test_regime_map_is_high_for_cone_heavy_fields_and_low_for_white_noise():
    shape = (32, 32, 32)
    model = make(dict(rho=2.0, w=[1.0, 1.0, 1.0], c=0.0))
    mask = torch.ones(1, 1, *shape)
    weight = torch.ones_like(mask)
    transform, _, mass = model._geometry(shape, torch.device("cpu"))
    assert len(mass) == 3 and all(m > 0 for m in mass)

    def regime_of(field):
        comps = torch.stack([torch.fft.ifftn(s, dim=(-3, -2, -1), norm="ortho").real[:, 0] for s in transform.analysis(field)], 1)
        return model.regime(comps, mass)[1].mean().item()

    white = cone_heavy(shape, 0.0)
    assert regime_of(cone_heavy(shape, 6.0)) > regime_of(white) + 0.2


def test_every_regime_parameter_gets_gradient_and_old_state_dicts_still_load():
    torch.manual_seed(1)
    model = make(dict(rho=2.0, g0=[0.9] * 7, g1=[0.2] * 7, w=[0.5, 0.5, 0.5], c=0.1))
    with torch.no_grad():
        model.head.weight.normal_(0, 0.1)
    field, mask, weight = inputs(SHAPE)
    model(field, mask, weight).square().sum().backward()
    for name in ("g0", "g1", "w", "c"):
        grad = getattr(model.regime, name).grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0, name
    plain = make()
    plain.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith("regime") and not k.startswith("lift")}, strict=False)


def test_regime_filter_rejects_wrong_sizes_and_plain_models():
    with pytest.raises(ValueError, match="one gain per segment"):
        RegimeFilter(["scale1/band0", "scale1/band1", "coarse"], g0=[1.0])
    with pytest.raises(ValueError, match="one weight"):
        RegimeFilter(["scale1/band0", "scale1/band1", "coarse"], w=[1.0, 2.0])
    with pytest.raises(ValueError, match="plain"):
        PhaseCleaner((1, 1, 1), dipolelet=DIPOLELET, plain=True, regime={}, **SMALL)


def test_regime_model_runs_on_another_shape_and_checkpointing_matches():
    torch.manual_seed(2)
    a = make(dict(rho=2.0, g0=[0.9] * 7, g1=[0.3] * 7, w=[0.5] * 3, c=0.0))
    with torch.no_grad():
        a.head.weight.normal_(0, 0.1)
    b = make(dict(rho=2.0, g0=[0.9] * 7, g1=[0.3] * 7, w=[0.5] * 3, c=0.0), checkpoint=True)
    b.load_state_dict(a.state_dict())
    field, mask, weight = inputs((20, 24, 22))
    assert torch.allclose(a(field, mask, weight), b(field, mask, weight), atol=1e-5)


@pytest.mark.parametrize("family", REAL_NOISE)
def test_noisy_real_residual_families(volume, family):
    a, info = draw(family, volume, seed=3, snr_low=40, snr_high=100)
    b, _ = draw(family, volume, seed=3, snr_low=40, snr_high=100)
    assert torch.isfinite(a).all() and torch.equal(a, b) and info["family"] == family and "snr" in info
    assert not torch.equal(a, draw(family, volume, seed=4, snr_low=40, snr_high=100)[0])
    base = volume.sti if family == "sti_noise" else volume.clean + volume.offdiag
    clean_limit, _ = draw(family, volume, seed=3, snr_low=1e7, snr_high=1e7)    # no noise: the structured residual itself
    assert torch.allclose(clean_limit * volume.mask, base * volume.mask, atol=1e-4)
    assert float(((a - base) * volume.mask).abs().sum()) > 0        # and the noise comes on top at SNR 40-100
    assert family in OBJECT_BOUND


def test_regime_run_trains_selects_and_saves_a_loadable_model(roots, tmp_path):  # noqa: F811
    rc1, rc2 = roots
    cfg = config(rc1, rc2, tmp_path / "run", 4, weights={"noise": 1.0, "clean": 0.25, "sti_noise": 0.5, "offdiag_noise": 0.5})
    cfg["dipolelet"] = SPLIT
    cfg["model"]["regime"] = dict(rho=2.0, g0=[1.0] * 7, g1=[0.5] * 7, w=[0.0, 0.0, 1.0], c=0.0)
    cfg["training"]["eval_every"] = 2
    summary = run(cfg)
    out = tmp_path / "run"
    assert (out / "DONE").exists() and summary["steps"] == 4
    table = rows(out / "eval.csv")
    assert {"sti_noise_test", "offdiag_noise_test", "clean_test", "sel_trained"} <= set(table[0])
    state = torch.load(out / "best.pt", weights_only=False)
    model = PhaseCleaner(state["voxel_size"], input_scale=state["input_scale"], dipolelet=state["config"]["dipolelet"], **state["config"]["model"])
    model.load_state_dict(state["model"])
