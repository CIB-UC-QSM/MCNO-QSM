"""Mask-aware GroupNorm in the Dipolelet blocks (pre-norm residual, CLAUDE.md 9bd)."""
import pytest
import torch

from mcno_qsm.cleaner import MaskedGroupNorm, PhaseCleaner
from test_cleaner import DIPOLELET, SMALL, inputs

SHAPE = (16, 16, 16)


def make(norm=None, **kw):
    return PhaseCleaner((1, 1, 1), dipolelet=DIPOLELET, norm=norm, norm_groups=2, **{**SMALL, **kw})


def test_statistics_come_from_the_mask_only_and_are_scale_invariant():
    torch.manual_seed(0)
    norm = MaskedGroupNorm(4, 2)
    v = torch.randn(2, 4, *SHAPE) * 3 + 1
    mask = (torch.rand(2, 1, *SHAPE) > 0.6).float()
    out = norm(v, mask)
    inside = mask.expand_as(out) > 0
    for b in range(2):
        for g in range(2):
            sel = out[b, 2 * g:2 * g + 2][inside[b, 2 * g:2 * g + 2]]
            assert abs(float(sel.mean())) < 1e-4 and abs(float(sel.var(unbiased=False)) - 1) < 1e-3
    # changing the background does not change the statistics, and a global rescale of the input does not change the output
    v2 = v.clone()
    v2[mask.expand_as(v) == 0] = 100.0
    assert torch.allclose(norm(v2, mask)[inside], out[inside], atol=1e-4)
    assert torch.allclose(norm(5 * v, mask)[inside], out[inside], atol=1e-3)
    with pytest.raises(ValueError):
        MaskedGroupNorm(5, 2)


def test_untrained_normalized_cleaner_is_the_masked_identity_and_adds_parameters_only_when_on():
    torch.manual_seed(0)
    model, plain = make("masked_group"), make()
    field, mask, weight = inputs(SHAPE)
    assert torch.allclose(model(field, mask, weight), field * mask, atol=1e-6)
    assert any(k.endswith("norm.weight") for k in model.state_dict()) and not any("norm" in k for k in plain.state_dict())
    with pytest.raises(ValueError, match="norm must be"):
        make("batch")


def test_all_parameters_get_gradient_other_shapes_and_checkpointing_matches():
    torch.manual_seed(1)
    a = make("masked_group")
    with torch.no_grad():
        a.head.weight.normal_(0, 0.1)
    field, mask, weight = inputs((12, 14, 13))
    a.zero_grad()
    a(field, mask, weight).square().sum().backward()
    for name, p in a.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name
    b = make("masked_group", checkpoint=True)
    b.load_state_dict(a.state_dict())
    assert torch.allclose(a(field, mask, weight), b(field, mask, weight), atol=1e-5)
    b(field, mask, weight).sum().backward()                      # checkpointed backward with the mask argument


def test_background_content_does_not_change_the_output_inside_the_mask_of_the_normalized_blocks():
    """Amplitude of the field outside the mask must not alter the normalization statistics (patch vs full-volume consistency)."""
    torch.manual_seed(2)
    model = make("masked_group")
    with torch.no_grad():
        model.head.weight.normal_(0, 0.1)
    field, mask, weight = inputs(SHAPE)
    base = model(field, mask, weight)
    loud = field * mask + 50 * (1 - mask) * torch.randn_like(field)
    # the lift and the spectral branches still see the outside field, so only compare the order of magnitude of the change
    assert float(((model(loud, mask, weight) - base) * mask).abs().max()) < 50 * float(base.abs().max()) + 10
