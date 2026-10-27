"""Full-volume Sim2 overfit experiment and reproducible artifacts."""

import argparse
import csv
import hashlib
import json
import platform
import random
import time
import tomllib
from pathlib import Path

import numpy as np
import torch

from .data import load_sim2, save_volume
from .metrics import QSMObjective, masked_nrmse
from .model import build_model
from .physics import DipoleForward
from .visualization import plot_history, plot_slices


def _seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def overfit(config: dict) -> dict:
    settings = config["training"]
    if settings["steps"] < 0 or settings["learning_rate"] <= 0:
        raise ValueError("steps must be nonnegative and learning_rate positive")
    if settings["snapshot_every"] < 1:
        raise ValueError("snapshot_every must be positive")
    output = Path(settings["output"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "history.csv").exists():
        raise FileExistsError(f"{output} already contains a run; choose a new --output")
    _seed(settings["seed"])
    requested = settings["device"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if requested == "auto" else torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu for checks")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # Avoid oversubscribing small CPU checks on high-core-count machines.
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 8))
    data = load_sim2(**config["data"])
    field = data.field.to(device) / data.input_scale
    ground_truth = data.ground_truth.to(device)
    mask = data.mask.to(device)
    shape = tuple(ground_truth.shape[-3:])
    b0 = tuple(config["data"]["b0"])
    model = build_model(shape, data.voxel_size, b0, **config["model"]).to(device)
    physics = DipoleForward(shape, data.voxel_size, b0, settings["padding"]).to(device)
    objective = QSMObjective(physics, ground_truth, mask,
                             settings["physics_weight"], settings["l1_weight"])
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    metadata = {
        "config": config, "shape": shape, "voxel_size": data.voxel_size,
        "affine": data.reference.affine.tolist(), "input_scale": data.input_scale,
        "device": str(device), "python": platform.python_version(), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "parameters": sum(p.numel() for p in model.parameters()),
        "depth": config["model"].get("depth", 1),
        "segments": model.transform.segment_names,
        "data_sha256": {k: _file_hash(config["data"][k]) for k in ("field", "ground_truth", "mask")},
        "reference_commit": "cf98d579bbd5c100c8698a81ccd630f4055568a1",
        "assumptions": {"field_unit": "ppm", "b0_coordinates": "image axes", "outside_mask_chi": 0},
        "metric": "norm((prediction-ground_truth)[mask])/norm(ground_truth[mask])",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    np.save(output / "simulated_field.npy", objective.target_field.cpu().numpy()[0, 0])
    plot_slices(data.ground_truth[0, 0], output / "ground_truth.png", title="Sim2 ground truth",
                voxel_size=data.voxel_size)
    history = []
    best = float("inf")
    best_step = 0
    started = time.perf_counter()

    def artifacts(prediction, prefix):
        prediction_cpu = prediction.detach().cpu()[0, 0]
        truth_cpu = data.ground_truth[0, 0]
        support_cpu = data.mask[0, 0]
        plot_slices(prediction_cpu, output / f"{prefix}.png", title=f"Sim2 {prefix}",
                    ground_truth=truth_cpu, mask=support_cpu, voxel_size=data.voxel_size)
        difference = (prediction_cpu - truth_cpu) * support_cpu
        plot_slices(difference, output / f"{prefix}_difference.png", title="Prediction − GT (masked)",
                    difference=True, voxel_size=data.voxel_size)
        save_volume(output / f"{prefix}.nii.gz", prediction_cpu, data.reference)
        save_volume(output / f"{prefix}_difference.nii.gz", difference, data.reference)

    print(f"device={device} shape={shape} depth={metadata['depth']} "
          f"segments_per_layer={len(model.transform.segment_names)} parameters={metadata['parameters']}", flush=True)
    with (output / "history.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", "loss", "physics", "l1", "nrmse", "seconds"])
        writer.writeheader()
        for step in range(settings["steps"] + 1):
            # Row zero is initialization; subsequent rows are AFTER each update.
            model.eval()
            with torch.no_grad():
                prediction = model(field) * mask
                loss, physical, l1 = objective(prediction)
                score = masked_nrmse(prediction, ground_truth, mask).item()
            row = dict(step=step, loss=loss.item(), physics=physical.item(), l1=l1.item(),
                       nrmse=score, seconds=time.perf_counter() - started)
            if not all(np.isfinite(row[k]) for k in ("loss", "physics", "l1", "nrmse")):
                raise FloatingPointError(f"Nonfinite result at step {step}: {row}")
            history.append(row)
            writer.writerow(row)
            stream.flush()
            print(f"step={step:04d} loss={row['loss']:.6f} physics={row['physics']:.6f} "
                  f"l1={row['l1']:.6f} nrmse={score:.6f} seconds={row['seconds']:.1f}", flush=True)
            if score < best:
                best, best_step = score, step
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                            "step": step, "nrmse": score, "metadata": metadata}, output / "best.pt")
            if step % settings["snapshot_every"] == 0 or step == settings["steps"]:
                artifacts(prediction, "latest")
                plot_history(history, output / "nrmse.png")
            if step == settings["steps"]:
                artifacts(prediction, "final")
                break
            del prediction, loss, physical, l1
            model.train()
            optimizer.zero_grad(set_to_none=True)
            training_prediction = model(field) * mask
            training_loss, _, _ = objective(training_prediction)
            training_loss.backward()
            optimizer.step()
            del training_prediction, training_loss

    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    with torch.no_grad():
        artifacts(model(field) * mask, "best")
    summary = {"best_nrmse": best, "best_step": best_step,
               "depth": metadata["depth"],
               "initial_nrmse": history[0]["nrmse"], "final_nrmse": history[-1]["nrmse"],
               "steps": settings["steps"], "target_nrmse": settings["target_nrmse"],
               "target_reached": best <= settings["target_nrmse"],
               "seconds": time.perf_counter() - started, "device": str(device)}
    if device.type == "cuda":
        summary["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(device)
        summary["peak_gpu_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("overfit", help="Overfit the complete Sim2 volume")
    run.add_argument("--config", type=Path, default=Path("configs/sim2.toml"))
    run.add_argument("--steps", type=int)
    run.add_argument("--output", type=Path)
    run.add_argument("--device", choices=("auto", "cpu", "cuda"))
    run.add_argument("--depth", type=int, help="Number of complete CNN/dipolelet layers")
    clean = commands.add_parser("clean", help="Train the phase cleaner on RC2 patches (resumable)")
    clean.add_argument("--config", type=Path, default=Path("configs/sim2_clean.toml"))
    clean.add_argument("--steps", type=int)
    clean.add_argument("--output", type=Path)
    clean.add_argument("--device", choices=("auto", "cpu", "cuda"))
    clean.add_argument("--resume", action="store_true", help="continue from OUTPUT/last.pt")
    rc1 = commands.add_parser("rc1", help="Train the phase cleaner on RC1 + simulated artifacts (S4b, resumable)")
    rc1.add_argument("--config", type=Path, required=True)
    rc1.add_argument("--steps", type=int)
    rc1.add_argument("--output", type=Path)
    rc1.add_argument("--device", choices=("auto", "cpu", "cuda"))
    rc1.add_argument("--resume", action="store_true", help="continue from OUTPUT/last.pt")
    args = parser.parse_args()
    if args.command == "clean":
        from .cleaner_train import cli

        return cli(args)
    if args.command == "rc1":
        from .rc1_train import cli

        return cli(args)
    with args.config.open("rb") as stream:
        config = tomllib.load(stream)
    for name in ("steps", "output", "device"):
        value = getattr(args, name)
        if value is not None:
            config["training"][name] = str(value) if isinstance(value, Path) else value
    if args.depth is not None:
        config["model"]["depth"] = args.depth
    overfit(config)


if __name__ == "__main__":
    main()
