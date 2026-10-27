import copy

import nibabel as nib
import numpy as np
import pytest
import torch

from mcno_qsm.cleaner import PhaseCleaner, k_features
from mcno_qsm.cleaner_train import Sim2Source, run
from mcno_qsm.losses import cone_energy, cone_loss, cone_mask, phase_loss
from mcno_qsm.noise import calibrate_snr, simulate_dirty
from mcno_qsm.scoring import forward_field

SMALL = dict(channels=4, blocks=2, dilations=(1, 2), checkpoint=False)
DIPOLELET = dict(scales=2, scaling_law="static", base_angular_bands=2, delta=0.05, window="gaussian",
                 gaussian_width_factor=0.5, starlet="gaussian", starlet_sigma=0.5)


def inputs(shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    field = torch.randn(1, 1, *shape, generator=g)
    mask = (torch.rand(1, 1, *shape, generator=g) > 0.3).float()
    return field, mask, torch.rand(1, 1, *shape, generator=g)


def test_untrained_cleaner_is_the_masked_identity():
    model = PhaseCleaner((1, 1, 1), dipolelet=DIPOLELET, input_scale=0.5, **SMALL)
    field, mask, weight = inputs((12, 12, 12))
    assert torch.allclose(model(field, mask, weight), field * mask, atol=1e-6)


def test_cleaner_handles_other_shapes_and_every_parameter_gets_gradient():
    torch.manual_seed(0)
    model = PhaseCleaner((1, 1, 1), dipolelet=DIPOLELET, **SMALL)
    with torch.no_grad():
        model.head.weight.normal_(0, 0.1)
    for shape in ((12, 12, 12), (10, 14, 11)):
        field, mask, weight = inputs(shape)
        model.zero_grad()
        out = model(field, mask, weight)
        assert out.shape == field.shape
        out.square().sum().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None or not torch.isfinite(p.grad).all()]
    assert not missing
    # The zero-initialized last layer of the k-gain MLP learns first; its first layer follows once it moves.
    assert all(p.grad.abs().sum() > 0 for n, p in model.named_parameters() if "gain.2" in n)


def test_cleaner_is_deterministic_and_checkpointing_matches():
    torch.manual_seed(1)
    a = PhaseCleaner((1, 1, 1), dipolelet=DIPOLELET, **SMALL)
    with torch.no_grad():
        a.head.weight.normal_(0, 0.1)
    b = copy.deepcopy(a)
    b.use_checkpoint = True
    field, mask, weight = inputs((12, 12, 12))
    out_a, out_b = a(field, mask, weight), b(field, mask, weight)
    assert torch.equal(out_a, a(field, mask, weight))
    out_a.square().sum().backward()
    out_b.square().sum().backward()
    for (_, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.allclose(p.grad, q.grad, atol=1e-5)


def test_k_features_are_physical_and_finite():
    f = k_features((8, 10, 12), (1.0, 1.0, 1.0))
    assert f.shape == (5, 8, 10, 12) and torch.isfinite(f).all()
    assert f[0, 0, 0, 0] == 0 and f[1, 0, 0, 0] == 0


def test_losses():
    target = torch.randn(1, 1, 8, 8, 8)
    mask = torch.ones_like(target)
    assert phase_loss(target, target, mask)[0] == 0
    loss, l1, l2 = phase_loss(torch.zeros_like(target), target, mask)
    assert l2.item() == pytest.approx(1.0) and l1.item() == pytest.approx(1.0)
    cone = cone_mask((8, 8, 8), (1, 1, 1), 0.1)
    assert cone_loss(target, target, cone) == 0
    assert 0 <= cone_energy(target, cone).item() <= 1


def test_simulated_dirty_is_seeded_and_calibration_hits_target():
    clean = 0.01 * torch.randn(1, 1, 16, 16, 16)
    mag = torch.full_like(clean, 0.8)
    mask = torch.ones_like(clean)
    scale = 15.0
    a = simulate_dirty(clean, mag, 50.0, scale, torch.Generator().manual_seed(3))
    b = simulate_dirty(clean, mag, 50.0, scale, torch.Generator().manual_seed(3))
    assert torch.equal(a, b)
    high = simulate_dirty(clean, mag, 5000.0, scale, torch.Generator().manual_seed(3))
    assert (high - clean).abs().mean() < (a - clean).abs().mean()
    snr = calibrate_snr(clean, mag, mask, scale, target_rms=0.004)
    residual = simulate_dirty(clean, mag, snr, scale, torch.Generator().manual_seed(0)) - clean
    assert residual.square().mean().sqrt().item() == pytest.approx(0.004, rel=0.15)


def write_tiny_rc2(root):
    rng = np.random.default_rng(0)
    shape = (24, 26, 24)
    chi = np.zeros(shape, np.float32)
    chi[6:18, 6:20, 6:18] = 0.05 * rng.standard_normal((12, 14, 12)).astype(np.float32)
    mask = np.zeros(shape, np.float32)
    mask[5:19, 5:21, 5:19] = 1
    clean = forward_field(chi, None)
    frequency = clean * 42.58 * 7 + 0.3 * rng.standard_normal(shape).astype(np.float32)
    magnitude = (rng.random((*shape, 4)) * 0.5 + 0.5).astype(np.float32)
    affine = np.eye(4)
    root.mkdir()
    for name, array in (("Chi", chi), ("Frequency", frequency), ("Magnitude", magnitude),
                        ("MaskBrainExtracted", mask)):
        nib.save(nib.Nifti1Image(array, affine), root / f"{name}.nii")


def tiny_config(root, output, steps):
    return {
        "data": {"root": str(root), "field_strength": 7.0, "te": 0.008},
        "dipolelet": DIPOLELET,
        "model": dict(channels=4, blocks=1, dilations=[1], k_conditioning=True, checkpoint=False),
        "training": dict(steps=steps, learning_rate=0.01, weight_decay=0.0, grad_clip=1.0, seed=7,
                         device="cpu", patch=16, batch_size=1, p_synthetic=0.5, snr_jitter=0.3,
                         l1_weight=1.0, l2_weight=1.0, cone_weight=0.1, log_every=1, eval_every=3,
                         save_every=3, output=str(output)),
    }


def test_training_resume_reproduces_an_uninterrupted_run(tmp_path):
    root = tmp_path / "rc2"
    write_tiny_rc2(root)
    full = run(tiny_config(root, tmp_path / "full", 6))
    assert (tmp_path / "full" / "DONE").exists() and full["steps"] == 6

    config = tiny_config(root, tmp_path / "resumed", 6)
    assert run(config, stop_after=3)["interrupted"]
    assert (tmp_path / "resumed" / "last.pt").exists() and not (tmp_path / "resumed" / "DONE").exists()
    with pytest.raises(FileExistsError):
        run(config)
    run(config, resume=True)

    a = torch.load(tmp_path / "full" / "final.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "resumed" / "final.pt", weights_only=False)["model"]
    assert a.keys() == b.keys()
    for key in a:
        assert torch.allclose(a[key], b[key], atol=1e-6), key
    assert (tmp_path / "full" / "eval.csv").exists() and (tmp_path / "full" / "best_clean.nii.gz").exists()


def test_failed_run_leaves_marker(tmp_path):
    config = tiny_config(tmp_path / "missing", tmp_path / "bad", 2)
    with pytest.raises(Exception):
        run(config)
    assert (tmp_path / "bad" / "FAILED").exists()


HOLDOUT = dict(axis=2, train_fraction=0.7, gap=2)


def test_holdout_patches_never_touch_the_test_slab(tmp_path):
    write_tiny_rc2(tmp_path / "rc2")
    source = Sim2Source(tmp_path / "rc2", 7.0, 0.008, torch.device("cpu"), p_synthetic=0.5, holdout=HOLDOUT)
    assert source.train_range == (0, 17) and source.test_range == (19, 24)
    for tensor in (source.field, source.clean_full, source.target, source.mask, source.weight, source.magnitude):
        tensor[..., 17:] = float("nan")
    rng, generator = np.random.default_rng(0), torch.Generator().manual_seed(0)
    for _ in range(60):
        assert all(torch.isfinite(t).all() for t in source.patch(16, rng, generator))
    assert set(source.regions) == {"_train", "_test"}


def test_holdout_run_reports_train_and_test_columns(tmp_path):
    write_tiny_rc2(tmp_path / "rc2")
    config = tiny_config(tmp_path / "rc2", tmp_path / "out", 3)
    config["data"]["holdout"] = dict(axis=1, train_fraction=0.7, gap=1)  # the tiny mask reaches slices 19-20
    run(config)
    header = (tmp_path / "out" / "eval.csv").read_text().splitlines()[0].split(",")
    assert {"nrmse_real_train", "nrmse_real_test", "nrmse_synthetic_test"} <= set(header)


def test_local_dilated_conv_is_identity_at_init_and_learns():
    torch.manual_seed(2)
    base = PhaseCleaner((1, 1, 1), dipolelet=DIPOLELET, **SMALL)
    local = PhaseCleaner((1, 1, 1), dipolelet=DIPOLELET, block_dilations=(1, 2), **SMALL)
    local.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        base.head.weight.normal_(0, 0.1)
        local.head.weight.copy_(base.head.weight)
    field, mask, weight = inputs((12, 12, 12))
    assert torch.allclose(base(field, mask, weight), local(field, mask, weight), atol=1e-6)
    local(field, mask, weight).square().sum().backward()
    assert all(b.local.weight.grad.abs().sum() > 0 for b in local.blocks)
    assert all(b.local is None for b in base.blocks)
    with pytest.raises(ValueError):
        PhaseCleaner((1, 1, 1), dipolelet=DIPOLELET, block_dilations=(1,), **SMALL)


def test_unet_branch_is_silent_at_init_handles_shapes_and_can_be_ablated():
    torch.manual_seed(3)
    base = PhaseCleaner((1, 1, 1), dipolelet=DIPOLELET, **SMALL)
    hybrid = PhaseCleaner((1, 1, 1), dipolelet=DIPOLELET, unet_base=2, **SMALL)
    hybrid.load_state_dict(base.state_dict(), strict=False)
    assert base.unet is None
    with torch.no_grad():
        base.head.weight.normal_(0, 0.1)
        hybrid.head.weight.copy_(base.head.weight)
    for shape in ((12, 12, 12), (10, 14, 11)):
        field, mask, weight = inputs(shape)
        assert torch.allclose(base(field, mask, weight), hybrid(field, mask, weight), atol=1e-6)
    assert hybrid.last_branch_ratio == 0.0
    with torch.no_grad():
        hybrid.unet.out.weight.normal_(0, 0.1)
    on = hybrid(field, mask, weight)
    assert hybrid.last_branch_ratio > 0
    hybrid.unet_active = False
    assert torch.allclose(hybrid(field, mask, weight), base(field, mask, weight), atol=1e-6)
    assert not torch.allclose(on, base(field, mask, weight), atol=1e-6)
    hybrid.unet_active = True
    hybrid.zero_grad()
    hybrid(field, mask, weight).square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in hybrid.unet.parameters())


def test_unet_checkpointing_matches():
    torch.manual_seed(4)
    a = PhaseCleaner((1, 1, 1), dipolelet=DIPOLELET, unet_base=2, **SMALL)
    with torch.no_grad():
        a.head.weight.normal_(0, 0.1)
        a.unet.out.weight.normal_(0, 0.1)
    b = copy.deepcopy(a)
    b.use_checkpoint = True
    field, mask, weight = inputs((12, 12, 12))
    a(field, mask, weight).square().sum().backward()
    b(field, mask, weight).square().sum().backward()
    for (n, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert torch.allclose(pa.grad, pb.grad, atol=1e-5), n
