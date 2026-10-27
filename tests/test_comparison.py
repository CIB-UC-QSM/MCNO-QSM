import csv
import json

import nibabel as nib
import numpy as np
from scipy.io import savemat

from mcno_qsm.compare import report, run


def test_memory_depth_selection_uses_measured_peaks(tmp_path, monkeypatch):
    # CPU-only selection test: measured peaks deliberately differ from extrapolation.
    import mcno_qsm.compare as comparison

    dummy = tmp_path / "data"
    dummy.write_bytes(b"original")
    config = tmp_path / "config.toml"
    path = dummy.as_posix()
    config.write_text(f'[data]\nfield="{path}"\nground_truth="{path}"\nmask="{path}"\n')
    observed = []

    def subprocess(command, **kwargs):
        destination = command[command.index("--output") + 1]
        if "profile" in command:
            depth = int(command[command.index("--depth") + 1])
            observed.append(depth)
            # Extrapolation suggests 9, but measured depths >=4 need 5GB extra.
            peak = (2 + 6.4 * depth + (5 if depth >= 4 else 0)) * 10**9
            from pathlib import Path
            Path(destination).write_text(json.dumps({"depth": depth, "status": "ok",
                                                      "peak_reserved_bytes": peak}))
    monkeypatch.setattr(comparison.subprocess, "run", subprocess)
    monkeypatch.setattr(comparison, "report", lambda _: None)
    output = tmp_path / "comparison"
    run(config, output, steps=100)
    selection = json.loads((output / "selection.json").read_text())
    assert 8 in observed and 9 in observed
    assert selection["selected_depth"] == 8
    assert selection["depths"] == [1, 2, 3, 8]
    assert selection["measured_gb_decimal"] == 58.2


def test_comparison_report_uses_final_volumes_and_shared_ranges(tmp_path):
    shape = (6, 7, 8)
    gt = np.full(shape, 0.02, dtype=np.float32)
    nib.save(nib.Nifti1Image(gt, np.eye(4)), tmp_path / "gt.nii.gz")
    savemat(tmp_path / "mask.mat", {"mask_final": np.ones(shape)})
    savemat(tmp_path / "field.mat", {"phase": np.arange(np.prod(shape)).reshape(shape)})
    depths = [1, 2, 3, 8]
    (tmp_path / "selection.json").write_text(json.dumps({"depths": depths}))
    for depth in depths:
        folder = tmp_path / f"depth-{depth}"
        folder.mkdir()
        summary = {"depth": depth, "steps": 1, "initial_nrmse": 1, "final_nrmse": 0.5,
                   "best_nrmse": 0.4, "best_step": 0, "seconds": 1}
        (folder / "summary.json").write_text(json.dumps(summary))
        data = {"field": str(tmp_path / "field.mat"), "ground_truth": str(tmp_path / "gt.nii.gz"),
                "mask": str(tmp_path / "mask.mat")}
        (folder / "metadata.json").write_text(json.dumps({"config": {"data": data}}))
        nib.save(nib.Nifti1Image(gt * 0.5, np.eye(4)), folder / "final.nii.gz")
        with (folder / "history.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=["step", "nrmse"])
            writer.writeheader()
            writer.writerows([{"step": 0, "nrmse": 1}, {"step": 1, "nrmse": 0.5}])
    result = report(tmp_path)
    assert [r["depth"] for r in result["runs"]] == depths
    assert all((tmp_path / name).stat().st_size > 1000
               for name in ("nrmse_comparison.png", "reconstructions.png", "differences.png",
                            "depth-8.png", "depth-8-difference.png"))
