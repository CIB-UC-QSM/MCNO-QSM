import pytest
import torch

from mcno_qsm.dipolelets import SIM2_DIPOLELET, DipoleletTransform, image_components
from mcno_qsm.losses import segment_loss
from mcno_qsm.rc1_train import run
from test_rc1_train import config, roots  # noqa: F401


def _transform(shape):
    return DipoleletTransform(shape, (1.0, 1.0, 1.0), **SIM2_DIPOLELET, dtype=torch.float64)


def test_segment_loss_matches_the_definition_and_vanishes_at_the_target():
    torch.manual_seed(0)
    shape = (20, 22, 24)
    transform = _transform(shape)
    target = torch.randn(1, 1, *shape, dtype=torch.float64)
    prediction = target + 0.1 * torch.randn_like(target)
    mask = (torch.rand(1, 1, *shape) > 0.2).double()
    gains = [0.5 + 0.1 * i for i in range(10)]
    loss = segment_loss(prediction, target, mask, transform, gains)
    parts = image_components(transform, ((prediction - target) * mask)[0, 0])
    expected = sum(g**2 * float((p * mask[0, 0]).square().sum()) for g, p in zip(gains, parts)) ** 0.5 / float((target * mask).norm())
    assert float(loss) == pytest.approx(expected, rel=1e-9)
    assert float(segment_loss(target, target, mask, transform, gains)) < 1e-10
    with pytest.raises(ValueError):
        segment_loss(prediction, target, mask, transform, gains[:5])


def test_segment_loss_has_a_finite_gradient():
    shape = (16, 16, 16)
    transform = _transform(shape)
    target = torch.randn(1, 1, *shape, dtype=torch.float64)
    prediction = (target + 0.1 * torch.randn_like(target)).requires_grad_()
    segment_loss(prediction, target, torch.ones_like(target), transform, [1.0] * 10).backward()
    assert torch.isfinite(prediction.grad).all() and prediction.grad.abs().sum() > 0


def test_trainer_logs_and_requires_the_gains(roots, tmp_path):  # noqa: F811
    rc1, rc2 = roots
    cfg = config(rc1, rc2, tmp_path / "seg", 3)
    cfg["training"].update(segment_weight=1.0, segment_gains=[1.0] * 5)   # the test Dipolelet has 5 segments
    run(cfg)
    header = (tmp_path / "seg" / "history.csv").read_text().splitlines()[0].split(",")
    assert "seg" in header
    bad = config(rc1, rc2, tmp_path / "bad", 3)
    bad["training"]["segment_weight"] = 1.0
    with pytest.raises(ValueError, match="segment_gains"):
        run(bad)
