import csv
import json

import numpy as np
import pytest
import torch

from mcno_qsm.ahead_data import MixedSource, split_subjects, volume_from_arrays
from mcno_qsm.rc1_train import build_source, run
from test_rc1_train import config, roots  # noqa: F401  (roots is a fixture)


def write_fake_ahead(root, n=5, shape=(40, 48, 44), seed=0):
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for i in range(n):
        mask = np.zeros(shape, np.uint8)
        mask[6:-6, 6:-6, 6:-6] = 1
        chi = (rng.normal(0, 0.03, shape) * mask).astype(np.float16)
        weight = (0.26 * mask).astype(np.float16)
        np.savez_compressed(root / f"sub-{i:04d}.npz", chi=chi, weight=weight, mask=mask,
                            voxel_size=np.ones(3, np.float32), summary=np.array("{}"))
    return root


def ahead_config(rc1, rc2, ahead, output, steps, fraction=0.5):
    cfg = config(rc1, rc2, output, steps, weights={"noise": 1.0, "clean": 0.25})
    cfg["data"]["ahead"] = {"root": str(ahead), "fraction": fraction, "val_subjects": 1, "spare_subjects": 1, "split_seed": 0,
                            "eval_subjects": 1, "crop": 32}
    return cfg


def test_split_is_disjoint_deterministic_and_complete():
    names = [f"sub-{i:04d}" for i in range(20)]
    a = split_subjects(names, 3, 2, seed=1)
    assert a == split_subjects(names, 3, 2, seed=1) and a != split_subjects(names, 3, 2, seed=2)
    train, val, spare = a
    assert (len(train), len(val), len(spare)) == (15, 3, 2)
    assert sorted(train + val + spare) == names and not (set(train) & set(val)) and not (set(val) & set(spare))
    with pytest.raises(ValueError):
        split_subjects(names[:3], 2, 1)


def test_volume_is_a_padded_cube_with_the_clean_field_of_its_chi():
    chi = np.random.default_rng(0).normal(0, 0.03, (20, 24, 22)).astype(np.float32)
    mask = np.ones_like(chi)
    v = volume_from_arrays(chi, 0.26 * mask, mask, (1.0, 1.0, 1.0), 10.0, "cpu", margin=8)
    assert v.shape == (32, 32, 32) and v.mask.sum() == mask.sum()
    assert torch.allclose(v.clean.sum(), torch.tensor(0.0), atol=1e-3 * float(v.clean.abs().sum()))   # D has zero mean


def test_batches_share_one_voxel_size_per_step(roots, tmp_path):  # noqa: F811
    rc1, rc2 = roots
    ahead = write_fake_ahead(tmp_path / "ahead")
    rng, generator = np.random.default_rng(0), torch.Generator().manual_seed(0)
    only_ahead = build_source(ahead_config(rc1, rc2, ahead, tmp_path / "o", 1, fraction=1.0), "cpu")
    assert isinstance(only_ahead, MixedSource)
    field, mask, weight, target, voxel = only_ahead.next_batch(16, rng, generator, 2)
    assert voxel == (1.0, 1.0, 1.0) and field.shape == mask.shape == weight.shape == target.shape == (2, 1, 16, 16, 16)
    only_rc1 = build_source(ahead_config(rc1, rc2, ahead, tmp_path / "o", 1, fraction=0.0), "cpu")
    assert only_rc1.next_batch(16, rng, generator, 2)[-1] == only_rc1.voxel_size
    assert only_ahead.train_names and not set(only_ahead.train_names) & set(only_ahead.val_names)


def test_ahead_patches_leave_identity_untouched(roots, tmp_path):  # noqa: F811
    rc1, rc2 = roots
    cfg = ahead_config(rc1, rc2, write_fake_ahead(tmp_path / "ahead"), tmp_path / "o", 1, fraction=1.0)
    cfg["artifacts"]["weights"] = {"clean": 1.0}
    source = build_source(cfg, "cpu")
    field, mask, weight, target, _ = source.next_batch(16, np.random.default_rng(1), torch.Generator().manual_seed(1), 2)
    assert torch.allclose(field * mask, target, atol=1e-6)         # dirty = clean inside the mask


def test_run_logs_ahead_validation_and_records_the_split(roots, tmp_path):  # noqa: F811
    rc1, rc2 = roots
    out = tmp_path / "run"
    run(ahead_config(rc1, rc2, write_fake_ahead(tmp_path / "ahead"), out, 6, fraction=0.5))
    with open(out / "eval.csv", newline="") as stream:
        table = list(csv.DictReader(stream))
    assert {"ahead_noise", "ahead_noise_in", "ahead_clean", "sel_trained", "sim2_nrmse"} <= set(table[0])
    meta = json.loads((out / "metadata.json").read_text())
    assert len(meta["ahead"]["train"]) == 3 and len(meta["ahead"]["validation"]) == 1 and len(meta["ahead"]["spare"]) == 1
