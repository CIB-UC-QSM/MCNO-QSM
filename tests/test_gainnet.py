"""Dipole-let gain network and its cascade with the frozen noise network (CLAUDE.md 9az)."""
import pytest
import torch

from mcno_qsm.cleaner import PhaseCleaner
from mcno_qsm.dipolelets import DipoleletTransform
from mcno_qsm.gainnet import GainCascade, GainNet, UNet3d
from mcno_qsm.rc1_train import run
from test_cleaner import DIPOLELET, SMALL, inputs
from test_rc1_train import config, roots  # noqa: F401  (roots is a fixture)

SPLIT = {**DIPOLELET, "coarse_bands": 3}       # scales=2 x 2 bands + split coarse (3) = 7 segments
SHAPE = (24, 24, 24)
SEGMENTS = 7
G1 = [0.0, 0.3, 0.0, 0.7, 0.2, 0.8, 1.0]
GAIN = dict(g1=G1, w=[1.0, 1.0, 1.0], c=0.0, rho=2.0, base=4, checkpoint=False)


def make_gain(**kw):
    return GainNet((1, 1, 1), dipolelet=SPLIT, input_scale=0.5, **{**GAIN, **kw})


def stage_b():
    torch.manual_seed(3)
    model = PhaseCleaner((1, 1, 1), dipolelet=DIPOLELET, input_scale=0.5, **SMALL)
    with torch.no_grad():
        model.head.weight.normal_(0, 0.1)           # a non-trivial B
    return model


def reference_stage_a(model, field, mask, eps=1e-3):
    """The passive regime-gated linear filter written out with the transform (no U-Net)."""
    transform, mass = model._geometry(field.shape[-3:], field.device)
    x = field * mask / model.input_scale
    comps = torch.stack([torch.fft.ifftn(s, dim=(-3, -2, -1), norm="ortho").real[:, 0] for s in transform.analysis(x)], 1)
    _, r = model.regime(comps, mass)
    g = (1 - r * (1 - model.regime.g1.view(1, -1, 1, 1, 1))).clamp(eps, 1 - eps)
    return mask * (x - ((1 - g) * comps).sum(1, keepdim=True)) * model.input_scale


def test_init_equals_the_passive_regime_gated_filter():
    model = make_gain()
    field, mask, weight = inputs(SHAPE)
    out = model(field, mask, weight)
    assert model.segments == SEGMENTS
    assert torch.allclose(out, reference_stage_a(model, field, mask), atol=1e-5)
    # with every g1 = 1 the filter is the masked identity
    ident = make_gain(g1=[1.0] * SEGMENTS)
    assert torch.allclose(ident(field, mask, weight), field * mask, atol=1e-2 * float(field.abs().max()))


def test_gains_are_passive_and_the_removal_never_exceeds_the_segments():
    torch.manual_seed(0)
    model = make_gain()
    with torch.no_grad():
        model.unet.out.weight.normal_(0, 1.0)
        model.unet.out.bias.normal_(0, 3.0)
    field, mask, weight = inputs(SHAPE)
    out = model(field, mask, weight)
    g = model.last_gain
    assert g.shape == (1, SEGMENTS, *SHAPE) and float(g.min()) > 0 and float(g.max()) < 1
    transform, _ = model._geometry(SHAPE, field.device)
    comps = torch.stack([torch.fft.ifftn(s, dim=(-3, -2, -1), norm="ortho").real[:, 0]
                         for s in transform.analysis(field * mask / model.input_scale)], 1)
    removed = (field * mask - out) / model.input_scale
    assert float(removed.abs().max()) <= float(comps.abs().sum(1).max()) + 1e-5
    assert torch.equal(out * (1 - mask), torch.zeros_like(out))


def test_only_the_unet_trains_and_all_of_it_gets_gradient():
    torch.manual_seed(1)
    model = make_gain()
    with torch.no_grad():
        model.unet.out.weight.normal_(0, 0.3)
    field, mask, weight = inputs(SHAPE)
    (model(field, mask, weight) - field * mask * 0.5).square().sum().backward()
    assert all(not p.requires_grad for p in model.regime.parameters())
    for name, p in model.unet.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    assert sum(float(p.grad.abs().sum()) for p in model.unet.parameters()) > 0


def test_other_shapes_and_checkpointing_match():
    torch.manual_seed(2)
    a = make_gain()
    with torch.no_grad():
        a.unet.out.weight.normal_(0, 0.3)
    b = make_gain(checkpoint=True)
    b.load_state_dict(a.state_dict())
    field, mask, weight = inputs((20, 24, 22))
    field.requires_grad_(False)
    fa, fb = a(field, mask, weight), b(field, mask, weight)
    assert torch.allclose(fa, fb, atol=1e-5)
    fb.square().sum().backward()                     # checkpointed backward runs


def test_rejects_non_passive_or_wrong_sized_gains():
    with pytest.raises(ValueError, match="passive gains"):
        make_gain(g1=[1.2] + G1[1:])
    with pytest.raises(ValueError, match="passive gains"):
        make_gain(g1=G1[:-1])


def test_unet_is_silent_at_init():
    net = UNet3d(5, 3, 4)
    assert float(net(torch.randn(1, 5, 9, 10, 11)).abs().max()) == 0.0


def test_cascade_is_b_after_a_with_the_outside_field_kept_and_b_stays_frozen(tmp_path):
    b = stage_b()
    cascade = GainCascade((1, 1, 1), stage_b=b, input_scale=0.5, dipolelet=SPLIT, gain=GAIN)
    field, mask, weight = inputs(SHAPE)
    cascade(field, mask, weight)
    assert cascade.gain.last_gain is None and make_gain().keep_gain        # the full-volume gain map is not kept in the cascade
    reference = b(reference_stage_a(cascade.gain, field, mask) + field * (1 - mask), mask, weight)
    assert torch.allclose(cascade(field, mask, weight), reference, atol=1e-5)
    before = {k: v.clone() for k, v in b.state_dict().items()}
    with torch.no_grad():
        cascade.gain.unet.out.weight.normal_(0, 0.3)
    cascade.train()
    assert not cascade.stage_b.training
    cascade(field, mask, weight).square().sum().backward()
    assert all(p.grad is None for p in cascade.stage_b.parameters())
    assert all(torch.equal(before[k], v) for k, v in b.state_dict().items())


def test_cascade_from_config_loads_the_checkpoint(tmp_path):
    b = stage_b()
    path = tmp_path / "b.pt"
    torch.save({"model": b.state_dict(), "voxel_size": (1.0, 1.0, 1.0), "input_scale": 0.5,
                "config": {"dipolelet": DIPOLELET, "model": {**SMALL, "checkpoint": False}}}, path)
    model_config = dict(kind="gain_cascade", stage_b=str(path), **GAIN)
    cascade = GainCascade.from_config((1, 1, 1), 0.5, SPLIT, model_config)
    field, mask, weight = inputs(SHAPE)
    assert torch.allclose(cascade.stage_b(field, mask, weight), b(field, mask, weight), atol=1e-6)
    with pytest.raises(ValueError, match="gain_cascade"):
        GainCascade.from_config((1, 1, 1), 0.5, SPLIT, {**model_config, "kind": "other"})


def test_gain_cascade_run_trains_only_the_gain_net_and_selects_on_rc1(roots, tmp_path):  # noqa: F811
    rc1, rc2 = roots
    b = stage_b()
    path = tmp_path / "b.pt"
    torch.save({"model": b.state_dict(), "voxel_size": (1.0, 1.0, 1.0), "input_scale": 0.5,
                "config": {"dipolelet": DIPOLELET, "model": {**SMALL, "checkpoint": False}}}, path)
    cfg = config(rc1, rc2, tmp_path / "run", 4, weights={"noise": 1.0, "clean": 0.25})
    cfg["dipolelet"] = SPLIT
    cfg["model"] = dict(kind="gain_cascade", stage_b=str(path), **{**GAIN, "checkpoint": True})
    summary = run(cfg)
    assert summary["steps"] == 4 and (tmp_path / "run" / "best.pt").exists()
    state = torch.load(tmp_path / "run" / "final.pt", weights_only=False)["model"]
    for k, v in b.state_dict().items():
        assert torch.equal(state[f"stage_b.{k}"], v), k          # B never moved
    assert any(float(v.abs().max()) > 0 for k, v in state.items() if k == "gain.unet.out.weight")


def test_neutralize_replaces_channels_by_their_median_and_rejects_unknown_names():
    torch.manual_seed(4)
    model = make_gain()
    with torch.no_grad():
        model.unet.out.weight.normal_(0, 0.5)
    field, mask, weight = inputs(SHAPE)
    plain = model(field, mask, weight).clone()
    model.neutralize = ("amp", "phi")
    changed = model(field, mask, weight)
    assert not torch.allclose(plain, changed, atol=1e-6)
    model.neutralize = ("weight",)                       # weight enters only through the U-Net: a constant weight gives another output too
    assert torch.isfinite(model(field, mask, weight)).all()
    model.neutralize = ()
    assert torch.allclose(model(field, mask, weight), plain, atol=1e-6)
    model.neutralize = ("bogus",)
    with pytest.raises(ValueError, match="cannot neutralize"):
        model(field, mask, weight)
