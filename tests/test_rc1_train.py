import csv

import pytest
import torch

from mcno_qsm.rc1_train import build_source, run
from test_cleaner import DIPOLELET, write_tiny_rc2
from test_rc1 import write_fake_rc1


def config(rc1, rc2, output, steps, weights=None):
    return {
        "data": {"rc1": str(rc1), "rc2": str(rc2), "field_strength": 7.0,
                 "holdout": {"axis": 0, "train_fraction": 0.7, "gap": 2}},
        "artifacts": {"snr_low": 40.0, "snr_high": 100.0, "weights": weights or {"combo": 2.0, "noise": 1.0}},
        "objects": {"probability": 0.5, "max_angle_deg": 5.0, "scale": [1.0, 1.0]},
        "dipolelet": DIPOLELET,
        "model": dict(channels=4, blocks=1, dilations=[1], k_conditioning=True, checkpoint=True),
        "training": dict(steps=steps, learning_rate=0.01, weight_decay=0.0, grad_clip=1.0, seed=7, device="cpu",
                         patch=16, batch_size=2, l1_weight=1.0, l2_weight=1.0, cone_weight=0.0, log_every=1,
                         eval_every=3, save_every=3, output=str(output)),
    }


@pytest.fixture()
def roots(tmp_path):
    rc1, rc2 = tmp_path / "rc1", tmp_path / "rc2"
    rc1.mkdir()
    write_fake_rc1(rc1)
    write_tiny_rc2(rc2)
    return rc1, rc2


def rows(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def test_rc1_run_selects_on_rc1_only_and_monitors_sim2(roots, tmp_path):
    rc1, rc2 = roots
    summary = run(config(rc1, rc2, tmp_path / "run", 6))
    out = tmp_path / "run"
    for name in ("DONE", "best.pt", "final.pt", "best_clean.nii.gz", "metadata.json", "eval.csv", "history.csv"):
        assert (out / name).exists(), name
    table = rows(out / "eval.csv")
    assert [int(r["step"]) for r in table] == [3, 6]
    assert {"sim2_nrmse", "sim2_nrmse_input", "sel_trained", "combo_test", "combo_in_test"} <= set(table[0])
    # The chosen step minimizes the RC1 selection metric, whatever Sim2 did.
    best_row = min(table, key=lambda r: float(r["sel_trained"]))
    assert summary["best_step"] == int(best_row["step"])
    assert "monitoring only" in (out / "metadata.json").read_text()


def test_selection_ignores_sim2(roots, tmp_path):
    rc1, rc2 = roots
    with_sim2 = run(config(rc1, rc2, tmp_path / "a", 6))
    no_sim2_cfg = config(rc1, rc2, tmp_path / "b", 6)
    no_sim2_cfg["data"].pop("rc2")
    without = run(no_sim2_cfg)
    assert with_sim2["best_step"] == without["best_step"] and not (tmp_path / "b" / "best_clean.nii.gz").exists()
    a = torch.load(tmp_path / "a" / "final.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "b" / "final.pt", weights_only=False)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a)  # Sim2 never touches training


def test_resume_reproduces_an_uninterrupted_run(roots, tmp_path):
    rc1, rc2 = roots
    run(config(rc1, rc2, tmp_path / "full", 6))
    cfg = config(rc1, rc2, tmp_path / "resumed", 6)
    assert run(cfg, stop_after=3)["interrupted"]
    with pytest.raises(FileExistsError):
        run(cfg)
    run(cfg, resume=True)
    a = torch.load(tmp_path / "full" / "final.pt", weights_only=False)["model"]
    b = torch.load(tmp_path / "resumed" / "final.pt", weights_only=False)["model"]
    assert all(torch.allclose(a[k], b[k], atol=1e-6) for k in a)


def test_real_psi_only_runs_never_rotate_the_head(roots):
    rc1, _ = roots
    cfg = config(rc1, "unused", "unused", 1, weights={"sti": 1.0, "offdiag": 1.0})
    cfg["objects"]["probability"] = 1.0
    source = build_source(cfg, "cpu")
    import numpy as np
    rng, generator = np.random.default_rng(0), torch.Generator().manual_seed(0)
    infos = [source.patch(16, rng, generator)[-1] for _ in range(10)]
    assert not any(i["object_augmented"] for i in infos) and {i["family"] for i in infos} <= {"sti", "offdiag"}


def test_limit_vram_uses_a_device_index_and_the_env_cap(monkeypatch):
    # regression: torch.device("cuda") has no index; the cap must resolve one (current_device) instead of raising
    from types import SimpleNamespace

    import torch

    from mcno_qsm.scoring import limit_vram

    calls = []
    monkeypatch.setenv("MCNO_VRAM_GB", "10")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: SimpleNamespace(total_memory=32 * 2**30))
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda fraction, index: calls.append((fraction, index)))
    assert limit_vram(torch.device("cuda")) == 10 / 32
    assert calls == [(10 / 32, 0)]
    monkeypatch.delenv("MCNO_VRAM_GB")
    assert limit_vram(torch.device("cuda")) is None


def test_band_loss_option_trains_and_logs_a_band_column(roots, tmp_path):
    rc1, rc2 = roots
    cfg = config(rc1, rc2, tmp_path / "band", 4)
    cfg["training"]["band_weight"] = 1.0
    run(cfg)
    history = rows(tmp_path / "band" / "history.csv")
    assert "band" in history[0] and all(float(r["band"]) > 0 for r in history)
    plain = config(rc1, rc2, tmp_path / "plain", 4)
    run(plain)
    assert "band" not in rows(tmp_path / "plain" / "history.csv")[0]   # off by default: same columns as before


def test_identity_family_is_the_clean_field_and_counts_as_damage_in_selection(roots, tmp_path):
    rc1, rc2 = roots
    cfg = config(rc1, rc2, tmp_path / "ident", 6, weights={"noise": 1.0, "clean": 1.0})
    summary = run(cfg)
    table = rows(tmp_path / "ident" / "eval.csv")
    assert {"clean_test", "clean_in_test"} <= set(table[0])
    assert all(float(r["clean_in_test"]) == 0.0 for r in table)  # dirty = clean: nothing to remove
    for r in table:  # selection = mean of (noise ratio, clean damage), both finite
        ratio_noise = float(r["noise_test"]) / float(r["noise_in_test"])
        assert float(r["sel_trained"]) == pytest.approx((ratio_noise + float(r["clean_test"])) / 2, rel=1e-5)
    assert summary["best_step"] in (3, 6)


def test_identity_family_off_by_default(roots, tmp_path):
    rc1, rc2 = roots
    run(config(rc1, rc2, tmp_path / "plain", 3))
    assert "clean_test" not in rows(tmp_path / "plain" / "eval.csv")[0]
