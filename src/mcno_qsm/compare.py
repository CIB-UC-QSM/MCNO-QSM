"""Measure CUDA memory, choose a depth near a budget, and compare overfit runs.

Each probe/run uses a fresh subprocess so CUDA allocator history cannot leak
between depths. GB means 10**9 bytes; both allocated and reserved peaks are saved.
"""

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

from .data import load_sim2
from .metrics import QSMObjective
from .model import build_model
from .physics import DipoleForward
from .train import _file_hash, _seed
from .visualization import plot_history, plot_slices


def profile(config: dict, depth: int) -> dict:
    """Measure evaluation plus two complete Adam updates at native resolution."""
    if not torch.cuda.is_available():
        raise RuntimeError("Memory profiling requires CUDA")
    _seed(config["training"]["seed"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    data = load_sim2(**config["data"])
    field = data.field.cuda() / data.input_scale
    gt, mask = data.ground_truth.cuda(), data.mask.cuda()
    kwargs = dict(config["model"], depth=depth)
    model = build_model(tuple(gt.shape[-3:]), data.voxel_size, config["data"]["b0"], **kwargs).cuda()
    settings = config["training"]
    physics = DipoleForward(tuple(gt.shape[-3:]), data.voxel_size, config["data"]["b0"],
                            settings["padding"]).cuda()
    objective = QSMObjective(physics, gt, mask, settings["physics_weight"], settings["l1_weight"])
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    started = time.perf_counter()
    for _ in range(2):
        model.eval()
        with torch.no_grad():
            prediction = model(field) * mask
            terms = objective(prediction)
        del prediction, terms
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(field) * mask
        terms = objective(prediction)
        if not torch.isfinite(terms[0]):
            raise FloatingPointError(f"Nonfinite initialization at depth {depth}")
        terms[0].backward()
        optimizer.step()
        del prediction, terms
    torch.cuda.synchronize()
    return {"depth": depth, "status": "ok", "updates": 2,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            "seconds": time.perf_counter() - started,
            "parameters": sum(p.numel() for p in model.parameters()),
            "config": dict(config, model=kwargs)}


def _read_history(path):
    with path.open() as stream:
        return [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(stream)]


def report(output: Path) -> dict:
    """Render common-scale curves and grids from completed comparison runs."""
    selection = json.loads((output / "selection.json").read_text())
    depths = selection["depths"]
    summaries, histories, predictions, errors = [], [], [], []
    reference_gt = reference_mask = None
    for depth in depths:
        folder = output / f"depth-{depth}"
        summary = dict(json.loads((folder / "summary.json").read_text()), depth=depth)
        memory = next((p for p in selection.get("profiles", []) if p["depth"] == depth), {})
        summary["profile_peak_reserved_bytes"] = memory.get("peak_reserved_bytes")
        metadata = json.loads((folder / "metadata.json").read_text())
        data = load_sim2(**metadata["config"]["data"])
        gt, mask = data.ground_truth[0, 0].numpy(), data.mask[0, 0].numpy()
        if reference_gt is None:
            reference_gt, reference_mask = gt, mask
        elif not np.array_equal(gt, reference_gt) or not np.array_equal(mask, reference_mask):
            raise ValueError("Comparison runs must use identical ground truth and mask")
        histories.append(_read_history(folder / "history.csv"))
        # Re-render reused baselines with the current gray cmap and [0,1] curve.
        plot_history(histories[-1], folder / "nrmse.png")
        plot_slices(gt, folder / "ground_truth.png", title="Sim2 ground truth", voxel_size=data.voxel_size)
        for prefix in ("best", "latest", "final"):
            path = folder / f"{prefix}.nii.gz"
            if path.exists():
                volume = nib.load(path).get_fdata(dtype=np.float32)
                plot_slices(volume, folder / f"{prefix}.png", title=f"Sim2 {prefix}, {depth} layer(s)",
                            ground_truth=gt, mask=mask, voxel_size=data.voxel_size)
                plot_slices((volume - gt) * mask, folder / f"{prefix}_difference.png",
                            title=f"{depth} layer(s): prediction − GT", difference=True,
                            voxel_size=data.voxel_size)
        # Compare the same final optimizer step, not independently chosen steps.
        prediction = nib.load(folder / "final.nii.gz").get_fdata(dtype=np.float32)
        predictions.append(prediction)
        errors.append((prediction - gt) * mask)
        plot_slices(prediction, output / f"depth-{depth}.png",
                    title=f"{depth} layer(s), step {summary['steps']}", ground_truth=gt, mask=mask,
                    voxel_size=data.voxel_size)
        plot_slices(errors[-1], output / f"depth-{depth}-difference.png",
                    title=f"{depth} layer(s): prediction − GT", difference=True,
                    voxel_size=data.voxel_size)
        summaries.append(summary)
    expected_steps = {s["steps"] for s in summaries}
    if len(expected_steps) != 1 or any(len(h) != summaries[0]["steps"] + 1 for h in histories):
        raise ValueError("Comparison requires identical complete iteration budgets")
    fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
    for depth, history in zip(depths, histories):
        ax.plot([r["step"] for r in history], [r["nrmse"] for r in history],
                label=f"{depth} layer(s), final={history[-1]['nrmse']:.4f}")
    ax.set(xlabel="Optimizer updates", ylabel="Masked NRMSE", ylim=(0, 1), title="Sim2 depth comparison")
    if any(r["nrmse"] > 1 for history in histories for r in history):
        ax.text(0.99, 0.03, "Values > 1 are outside the displayed range", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(output / "nrmse_comparison.png", dpi=180)
    plt.close(fig)

    def grid(volumes, path, limits, difference=False):
        fig, axes = plt.subplots(len(depths), 3, figsize=(10, 3.3 * len(depths)),
                                 squeeze=False, layout="constrained")
        center = tuple(n // 2 for n in gt.shape)
        for row, (depth, volume, summary) in enumerate(zip(depths, volumes, summaries)):
            slices = (volume[center[0], :, :], volume[:, center[1], :], volume[:, :, center[2]])
            aspects = (data.voxel_size[2] / data.voxel_size[1], data.voxel_size[2] / data.voxel_size[0],
                       data.voxel_size[1] / data.voxel_size[0])
            for col, (image, aspect) in enumerate(zip(slices, aspects)):
                im = axes[row, col].imshow(image.T, origin="lower", cmap="gray", aspect=aspect,
                                           vmin=limits[0], vmax=limits[1])
                axes[row, col].set_title(f"Coronal | {depth} layer(s) | NRMSE={summary['final_nrmse']:.4f}" if col == 1
                                         else ("Sagittal", "", "Axial")[col])
                axes[row, col].axis("off")
        fig.suptitle(f"Sim2 {'prediction − GT' if difference else 'reconstruction'} | step {summaries[0]['steps']}")
        fig.colorbar(im, ax=axes.ravel().tolist(), label="ppm", shrink=0.8)
        fig.savefig(path, dpi=160)
        plt.close(fig)

    grid(predictions, output / "reconstructions.png", (-0.1, 0.1))
    grid(errors, output / "differences.png", (-0.03, 0.03), difference=True)
    result = {"memory_selection": selection, "runs": summaries}
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    with (output / "comparison.csv").open("w", newline="") as stream:
        keys = ("depth", "steps", "initial_nrmse", "final_nrmse", "best_nrmse", "best_step",
                "seconds", "peak_gpu_memory_bytes", "peak_gpu_reserved_bytes", "profile_peak_reserved_bytes")
        writer = csv.DictWriter(stream, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summaries)
    return result


def run(config_path: Path, output: Path, steps=100, target_gb=60.0, max_depth=24, baseline=None):
    if steps < 1 or not 0 < target_gb < float("inf") or max_depth < 3:
        raise ValueError("steps and target_gb must be positive, max_depth must be >=3")
    output.mkdir(parents=True, exist_ok=True)
    with config_path.open("rb") as stream:
        config = tomllib.load(stream)
    hashes = {key: _file_hash(config["data"][key]) for key in ("field", "ground_truth", "mask")}
    manifest = {"config": config, "data_sha256": hashes, "steps": steps, "target_gb": target_gb,
                "max_depth": max_depth, "torch": torch.__version__,
                "baseline": str(baseline.resolve()) if baseline else None}
    manifest_path = output / "experiment.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("Output belongs to a different experiment; choose a new folder")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if baseline and not (output / "depth-1").exists():
        metadata = json.loads((baseline / "metadata.json").read_text())
        summary = json.loads((baseline / "summary.json").read_text())

        def normalized(value):
            value = json.loads(json.dumps(value))
            value["model"].setdefault("depth", 1)
            value["training"].pop("output", None)
            value["training"]["device"] = "cuda"
            value["training"]["steps"] = steps
            return value

        if (summary["steps"] != steps or summary["device"] != "cuda"
                or metadata["torch"] != torch.__version__
                or metadata["data_sha256"] != hashes
                or normalized(metadata["config"]) != normalized(config)):
            raise ValueError("Baseline data, configuration, versions and budget must match")
        shutil.copytree(baseline, output / "depth-1")
    profiles_dir = output / "profiles"
    profiles_dir.mkdir(exist_ok=True)
    profiles = {}

    def measure(depth):
        if depth not in profiles:
            destination = profiles_dir / f"depth-{depth}.json"
            command = [sys.executable, "-m", "mcno_qsm.compare", "profile", "--config", str(config_path),
                       "--depth", str(depth), "--output", str(destination)]
            if not destination.exists():
                with destination.with_suffix(".log").open("w") as stream:
                    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            profiles[depth] = json.loads(destination.read_text())
            print("PROFILE " + json.dumps(profiles[depth]), flush=True)
        return profiles[depth]

    target = target_gb * 10**9
    first, second = measure(1), measure(2)
    if first["status"] != "ok" or second["status"] != "ok":
        raise RuntimeError("Cannot profile the one/two-layer baselines")
    slope = max(1, second["peak_reserved_bytes"] - first["peak_reserved_bytes"])
    estimate = min(max_depth, max(3, math.floor(1 + (target - first["peak_reserved_bytes"]) / slope)))
    measure(3)
    measure(estimate)
    # Measured allocator peaks, not linear extrapolation, decide the final depth.
    # Establish adjacent depths on opposite sides of the target.
    for _ in range(max_depth):
        good = {d: p for d, p in profiles.items() if p["status"] == "ok"}
        below = [d for d, p in good.items() if p["peak_reserved_bytes"] <= target]
        above = [d for d, p in profiles.items()
                 if p["status"] != "ok" or p["peak_reserved_bytes"] > target]
        lo, hi = max(below, default=0), min(above, default=max_depth + 1)
        if hi - lo <= 1:
            break
        candidate = lo + 1 if hi == max_depth + 1 else max(1, (lo + hi) // 2)
        if candidate > max_depth:
            break
        measure(candidate)
    valid = [p for p in profiles.values() if p["status"] == "ok"]
    selected = min(valid, key=lambda p: abs(p["peak_reserved_bytes"] - target))
    depths = sorted(set([1, 2, 3, selected["depth"]]))
    selection = {"target_gb_decimal": target_gb, "metric": "torch.cuda.max_memory_reserved",
                 "selected_depth": selected["depth"],
                 "measured_gb_decimal": selected["peak_reserved_bytes"] / 10**9,
                 "depths": depths, "profiles": sorted(profiles.values(), key=lambda p: p["depth"]),
                 "steps_per_model": steps, "composition": "direct, independent weights"}
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    print("SELECTION " + json.dumps(selection), flush=True)
    for depth in depths:
        destination = output / f"depth-{depth}"
        command = [sys.executable, "-m", "mcno_qsm.train", "overfit", "--config", str(config_path),
                   "--depth", str(depth), "--steps", str(steps), "--device", "cuda",
                   "--output", str(destination)]
        if not (destination / "summary.json").exists():
            with (output / f"depth-{depth}.log").open("w") as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print(f"COMPLETED depth={depth}", flush=True)
    report(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    probe = commands.add_parser("profile")
    probe.add_argument("--config", type=Path, required=True)
    probe.add_argument("--depth", type=int, required=True)
    probe.add_argument("--output", type=Path, required=True)
    experiment = commands.add_parser("run")
    experiment.add_argument("--config", type=Path, default=Path("configs/sim2.toml"))
    experiment.add_argument("--output", type=Path, default=Path("runs/depth-comparison"))
    experiment.add_argument("--steps", type=int, default=100)
    experiment.add_argument("--target-gb", type=float, default=60)
    experiment.add_argument("--max-depth", type=int, default=24)
    experiment.add_argument("--baseline", type=Path, help="Reuse a matching completed one-layer run")
    redraw = commands.add_parser("report")
    redraw.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "profile":
        with args.config.open("rb") as stream:
            config = tomllib.load(stream)
        try:
            result = profile(config, args.depth)
        except torch.cuda.OutOfMemoryError:
            result = {"depth": args.depth, "status": "oom"}
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    elif args.command == "run":
        run(args.config, args.output, args.steps, args.target_gb, args.max_depth, args.baseline)
    else:
        report(args.output)


if __name__ == "__main__":
    main()
