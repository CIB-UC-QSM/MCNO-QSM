import csv

import numpy as np
import torch

from mcno_qsm.artifacts import CALC, FAMILIES, ArtifactConfig, apply_overrides, dipole_field, make_dirty
from mcno_qsm.rc1_train import run
from test_rc1 import fake_root, volume  # noqa: F401  (fixtures)
from test_rc1_train import config, roots  # noqa: F401


def draw(volume, seed=0, **cfg):
    rng, generator = np.random.default_rng(seed), torch.Generator().manual_seed(seed)
    return make_dirty(CALC, volume, rng, generator, ArtifactConfig(snr_low=1e6, snr_high=1e6, **cfg))


def test_calc_shell_is_not_one_of_the_nine_monitored_families():
    assert CALC not in FAMILIES


def test_the_new_clean_field_contains_the_added_source_and_the_magnitude_dips(volume):  # noqa: F811
    dirty, info = draw(volume, calc_count=(1, 1), calc_positive_prob=0.0)
    added = info["clean"] - volume.clean
    assert float(added.abs().max()) > 0.02                                        # a strong dipole field was added
    assert torch.allclose(dipole_field(volume, torch.zeros_like(volume.clean)), torch.zeros_like(volume.clean))
    assert float(info["magnitude"].min()) < float(volume.magnitude[volume.mask > 0].min()) - 1e-6   # dropout dip inside the source
    assert (info["magnitude"] <= volume.magnitude + 1e-7).all()
    changed = apply_overrides(volume, info)
    assert torch.equal(changed.target, volume.mask * info["clean"])               # the training target follows the added source


def test_the_shell_keeps_only_a_fraction_of_the_true_field_and_outside_is_clean(volume):  # noqa: F811
    dirty, info = draw(volume, calc_count=(1, 1), calc_noise=(0.0, 0.0), calc_keep=(0.3, 0.3), calc_ppm=(1.5, 1.5), calc_radius=(3.0, 3.0))
    clean, mask = info["clean"], volume.mask > 0
    # the voxel of the largest |added field| inside the mask lies in the source / shell: dirty = keep * clean there (no noise: SNR 1e6)
    added = (info["clean"] - volume.clean).abs() * volume.mask
    index = np.unravel_index(int(torch.argmax(added)), clean.shape)
    ratio = float(dirty[index] / clean[index])
    assert 0.25 < ratio < 0.5                                                      # shell edge may be soft, centre keeps ~0.3
    assert torch.allclose(dirty[~mask], clean[~mask])                              # outside the mask the field stays noise-free


def test_calc_shell_is_deterministic_in_its_seeds(volume):  # noqa: F811
    a, _ = draw(volume, seed=3)
    b, _ = draw(volume, seed=3)
    c, _ = draw(volume, seed=4)
    assert torch.equal(a, b) and not torch.equal(a, c)


def test_trainer_runs_with_the_family_and_logs_it(roots, tmp_path):  # noqa: F811
    rc1, rc2 = roots
    cfg = config(rc1, rc2, tmp_path / "run", 3, weights={"noise": 1.0, "clean": 0.25, "calc_shell": 0.5})
    run(cfg)
    with open(tmp_path / "run" / "eval.csv", newline="") as stream:
        row = list(csv.DictReader(stream))[-1]
    assert {"calc_shell_test", "calc_shell_in_test", "clean_test"} <= set(row)
    assert np.isfinite(float(row["calc_shell_test"])) and float(row["calc_shell_in_test"]) > 0
