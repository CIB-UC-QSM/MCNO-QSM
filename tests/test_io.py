import nibabel as nib
import numpy as np
import pytest
import torch
from scipy.io import savemat

from mcno_qsm.data import load_sim2, save_volume
from mcno_qsm.metrics import masked_nrmse
from mcno_qsm.train import overfit
from mcno_qsm.visualization import plot_history, plot_slices


def test_data_crop_export_and_plots(tmp_path):
    shape = (6, 7, 8)
    gt = np.full(shape, 0.02, dtype=np.float32)
    affine = np.diag([0.9, 1.1, 1.3, 1])
    affine[:3, 3] = [4, 5, 6]
    nib.save(nib.Nifti1Image(gt, affine), tmp_path / "gt.nii.gz")
    field = np.arange(8 * 9 * 10, dtype=np.float32).reshape(8, 9, 10)
    savemat(tmp_path / "field.mat", {"phase": field})
    savemat(tmp_path / "mask.mat", {"mask_final": np.ones(shape)})
    data = load_sim2(tmp_path / "field.mat", tmp_path / "gt.nii.gz", tmp_path / "mask.mat",
                     crop_start=(1, 1, 1), field_scale=2)
    np.testing.assert_array_equal(data.field[0, 0], 2 * field[1:7, 1:8, 1:9])
    assert data.voxel_size == pytest.approx((0.9, 1.1, 1.3))
    save_volume(tmp_path / "output.nii.gz", data.ground_truth, data.reference)
    restored = nib.load(tmp_path / "output.nii.gz")
    np.testing.assert_array_equal(restored.get_fdata(), gt)
    np.testing.assert_allclose(restored.affine, affine)
    plot_slices(data.ground_truth[0, 0], tmp_path / "prediction.png",
                ground_truth=data.ground_truth[0, 0], mask=data.mask[0, 0])
    plot_slices(torch.zeros(shape), tmp_path / "difference.png", difference=True)
    plot_history([{"step": 0, "nrmse": 1}, {"step": 1, "nrmse": 0.5}], tmp_path / "curve.png")
    assert all((tmp_path / name).stat().st_size > 1000
               for name in ("prediction.png", "difference.png", "curve.png"))


@pytest.mark.parametrize("depth", [1, 2])
def test_training_artifacts_and_best_checkpoint(tmp_path, depth):
    import csv
    import json

    shape = (6, 7, 8)
    rng = np.random.default_rng(42)
    chi = (rng.normal(size=shape) * 0.03).astype(np.float32)
    nib.save(nib.Nifti1Image(chi, np.eye(4)), tmp_path / "gt.nii.gz")
    savemat(tmp_path / "field.mat", {"phase": rng.normal(size=shape) * 0.01})
    savemat(tmp_path / "mask.mat", {"mask_final": np.ones(shape)})
    output = tmp_path / "run"
    config = {
        "data": {"field": str(tmp_path / "field.mat"), "ground_truth": str(tmp_path / "gt.nii.gz"),
                 "mask": str(tmp_path / "mask.mat"), "b0": [0, 0, 1]},
        "model": {"depth": depth, "scales": 1, "angular_bands": [2], "width": 4, "checkpoint": True,
                  "chunk_size": 128},
        "training": {"steps": 2, "learning_rate": 0.001, "seed": 42, "device": "cpu",
                     "padding": 2, "physics_weight": 1, "l1_weight": 1, "target_nrmse": 0.1,
                     "snapshot_every": 2, "output": str(output)},
    }
    summary = overfit(config)
    assert summary["depth"] == depth
    with (output / "history.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(r["step"]) for r in rows] == [0, 1, 2]
    assert summary["best_nrmse"] == min(float(r["nrmse"]) for r in rows)
    exported = torch.tensor(nib.load(output / "best.nii.gz").get_fdata(dtype=np.float32))
    score = masked_nrmse(exported, torch.tensor(chi), torch.ones(shape, dtype=torch.bool)).item()
    assert score == pytest.approx(summary["best_nrmse"], rel=1e-6)
    checkpoint = torch.load(output / "best.pt", weights_only=False)
    assert checkpoint["step"] == summary["best_step"]
    assert json.loads((output / "summary.json").read_text()) == summary
    assert json.loads((output / "metadata.json").read_text())["data_sha256"]["field"]
    assert all((output / name).exists() for name in
               ("final.png", "final_difference.png", "best.nii.gz", "final_difference.nii.gz", "nrmse.png"))
    with pytest.raises(FileExistsError):
        overfit(config)
