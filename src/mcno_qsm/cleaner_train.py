"""Train the PhaseCleaner on RC2 (Sim2) patches; resumable, with periodic full-volume evaluation."""

import csv
import hashlib
import json
import os
import platform
import random
import time
import tomllib
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .cleaner import PhaseCleaner
from .losses import cone_energy, cone_loss, cone_mask, phase_loss
from .metrics import masked_nrmse
from .noise import calibrate_snr, simulate_dirty
from .rc2 import load_rc2, phase_scale
from .scoring import forward_field


def _tensor(x, device):
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)[None, None]


class Sim2Source:
    """Full-volume tensors plus random augmented patches, real or freshly simulated dirty input."""

    def __init__(self, root, field_strength, te, device, p_synthetic=0.5, snr_jitter=0.3, holdout=None):
        """`holdout` = {axis, train_fraction, gap}: patches and every statistic that touches the target come
        from the first `train_fraction` of the axis; `gap` slices are skipped; the rest is the test slab."""
        data = load_rc2(root, field_strength)
        self.voxel_size, self.affine, self.device = data["voxel_size"], data["affine"], device
        self.p_synthetic, self.snr_jitter = p_synthetic, snr_jitter
        self.phs_scale = phase_scale(te, field_strength)
        clean_full = forward_field(data["chi"], None, self.voxel_size)
        mask = data["mask"]
        self.shape = tuple(mask.shape)
        self.holdout = holdout
        self.axis, train_end, test_start = 0, self.shape[0], 0
        if holdout:
            self.axis = int(holdout["axis"])
            n = self.shape[self.axis]
            train_end = int(round(holdout["train_fraction"] * n))
            test_start = train_end + int(holdout["gap"])
            if not 0 < train_end < test_start < n:
                raise ValueError(f"empty train or test slab: train [0,{train_end}) test [{test_start},{n})")
        self.train_range = (0, train_end if holdout else self.shape[0])
        self.test_range = (test_start, self.shape[self.axis]) if holdout else None
        train = self._region(self.train_range) if holdout else np.ones(self.shape, bool)
        # Statistics (input scale, noise level) come from the training region only.
        self.scale = float((data["field"] * mask)[mask.astype(bool) & train].std())
        self.mask = _tensor(mask, device)
        self.weight = _tensor(data["weight"], device)
        self.field = _tensor(data["field"], device)
        self.clean_full = _tensor(clean_full, device)
        self.target = self.clean_full * self.mask
        # Magnitude of the echo closest to TE (the notebook uses echo 2 for the simulated draws).
        self.magnitude = _tensor(data["magnitude"][..., 1] / data["magnitude"].max(), device)
        train_mask = self.mask * _tensor(train, device)
        psi_rms = float((self.field - self.clean_full)[train_mask > 0].square().mean().sqrt())
        self.snr = calibrate_snr(self.clean_full, self.magnitude, train_mask, self.phs_scale, psi_rms)
        self.psi_rms = psi_rms
        self.regions = {"": self.mask}
        if holdout:
            self.regions = {"_train": train_mask,
                            "_test": self.mask * _tensor(self._region(self.test_range), device)}

    def _region(self, bounds):
        region = np.zeros(self.shape, bool)
        index = [slice(None)] * 3
        index[self.axis] = slice(*bounds)
        region[tuple(index)] = True
        return region

    def fixed_synthetic_field(self, seed=12345):
        generator = torch.Generator(device=self.device).manual_seed(seed)
        return simulate_dirty(self.clean_full, self.magnitude, self.snr, self.phs_scale, generator)

    def patch(self, size, rng, generator):
        """One augmented patch (field, mask, weight, target), each [1,1,P,P,P]."""
        limits = list(self.shape)
        limits[self.axis] = self.train_range[1]
        high = [n - size for n in limits]
        if min(high) < 0:
            raise ValueError(f"patch {size} does not fit the training region {limits}")
        for _ in range(30):
            origin = [int(rng.integers(0, h + 1)) for h in high]
            window = (slice(None), slice(None), *[slice(o, o + size) for o in origin])
            if self.mask[window].mean() > 0.05:
                break
        synthetic = rng.random() < self.p_synthetic
        snr = self.snr * (1 + self.snr_jitter * (2 * rng.random() - 1))
        flips = [d for d in (2, 3, 4) if rng.random() < 0.5]
        quarter_turns = int(rng.integers(0, 4))
        clean, mag = self.clean_full[window], self.magnitude[window]
        field = (simulate_dirty(clean, mag, snr, self.phs_scale, generator) if synthetic
                 else self.field[window])
        tensors = [field, self.mask[window], self.weight[window], self.target[window]]
        out = []
        for t in tensors:
            if flips:
                t = torch.flip(t, flips)
            out.append(torch.rot90(t, quarter_turns, (2, 3)))
        return out


def _file_hash(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _learning_rate(step, steps, base, warmup=100):
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return base * 0.5 * (1 + float(np.cos(np.pi * min(1.0, progress))))


def _append(path, fields, row):
    new = not path.exists()
    with path.open("a", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if new:
            writer.writeheader()
        writer.writerow(row)


def _save_last(path, model, optimizer, step, best, rng, generator, config):
    state = {
        "model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "best": best,
        "rng": {"numpy": rng.bit_generator.state, "generator": generator.get_state(),
                "torch": torch.get_rng_state(), "python": random.getstate(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
        "config": config,
    }
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def evaluate(model, source, cone, output, step, best, config):
    """Full-volume scores on the real dirty field and a fixed fresh-noise draw; saves the best cleaned field."""
    model.eval()
    row = {"step": step}
    cleaned = {}
    for name, field in (("real", source.field), ("synthetic", source.fixed_synthetic_field())):
        prediction = model(field, source.mask, source.weight)
        cleaned[name] = prediction
        for tag, region in source.regions.items():
            row[f"nrmse_{name}{tag}"] = masked_nrmse(prediction, source.target, region).item()
            row[f"nrmse_input_{name}{tag}"] = masked_nrmse(field * source.mask, source.target, region).item()
        row[f"cone_{name}"] = cone_energy(prediction, cone).item()
    row["cone_target"] = cone_energy(source.target, cone).item()
    # With a hold-out the best checkpoint is chosen on the training region only, never on the test slab.
    key = "nrmse_real_train" if source.holdout else "nrmse_real"
    improved = row[key] < best
    if improved:
        best = row[key]
        image = nib.Nifti1Image(cleaned["real"].cpu().numpy()[0, 0], source.affine)
        nib.save(image, output / "best_clean.nii.gz")
    _append(output / "eval.csv", list(row) + ["best"], {**row, "best": best})
    print("eval " + " ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
          flush=True)
    model.train()
    return best, improved


def run(config: dict, resume: bool = False, stop_after: int | None = None) -> dict:
    t = config["training"]
    output = Path(t["output"])
    last = output / "last.pt"
    if resume:
        if not last.exists():
            raise FileNotFoundError(f"nothing to resume: {last}")
    else:
        if (output / "history.csv").exists() or last.exists():
            raise FileExistsError(f"{output} already contains a run; choose a new --output or use --resume")
        output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if t["device"] == "auto" \
        else torch.device(t["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    try:
        torch.manual_seed(t["seed"])
        random.seed(t["seed"])
        source = Sim2Source(config["data"]["root"], config["data"]["field_strength"], config["data"]["te"],
                            device, t["p_synthetic"], t["snr_jitter"], config["data"].get("holdout"))
        model = PhaseCleaner(source.voxel_size, input_scale=source.scale, dipolelet=config["dipolelet"],
                             **config["model"]).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
        rng = np.random.default_rng(t["seed"])
        generator = torch.Generator(device=device).manual_seed(t["seed"])
        step, best = 0, float("inf")
        if resume:
            state = torch.load(last, map_location=device, weights_only=False)
            if state["config"]["model"] != config["model"] or state["config"]["dipolelet"] != config["dipolelet"]:
                raise ValueError("model or dipolelet configuration differs from the checkpoint")
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            rng.bit_generator.state = state["rng"]["numpy"]
            generator.set_state(state["rng"]["generator"].cpu())
            torch.set_rng_state(state["rng"]["torch"].cpu())
            random.setstate(state["rng"]["python"])
            if state["rng"]["cuda"] is not None and device.type == "cuda":
                torch.cuda.set_rng_state_all([s.cpu() for s in state["rng"]["cuda"]])
            step, best = state["step"], state["best"]
            print(f"resumed from step {step}", flush=True)
        else:
            meta = {
                "config": config, "shape": source.shape, "voxel_size": source.voxel_size,
                "input_scale": source.scale, "snr_calibrated": source.snr, "psi_rms_ppm": source.psi_rms,
                "holdout": {"axis": source.axis, "train": source.train_range, "test": source.test_range}
                if source.holdout else None,
                "parameters": sum(p.numel() for p in model.parameters()), "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "python": platform.python_version(), "torch": torch.__version__,
                "data_sha256": {n: _file_hash(Path(config["data"]["root"]) / n)
                                for n in ("Chi.nii", "Frequency.nii", "Magnitude.nii", "MaskBrainExtracted.nii")},
                "target": "mask * forward(Chi), correctly aligned dipole kernel, unpadded",
            }
            (output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        size = t["patch"]
        cone_cache = {}
        full_cone = cone_mask(source.shape, source.voxel_size, config["dipolelet"]["delta"], device=device)
        started = time.perf_counter()
        steps = t["steps"]
        fields = ["step", "loss", "l1", "l2", "cone_excess", "lr", "seconds"]
        model.train()
        while step < steps:
            lr = _learning_rate(step, steps, t["learning_rate"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = [source.patch(size, rng, generator) for _ in range(t["batch_size"])]
            field, mask, weight, target = (torch.cat(items) for items in zip(*batch))
            if size not in cone_cache:
                cone_cache[size] = cone_mask((size,) * 3, source.voxel_size, config["dipolelet"]["delta"],
                                             device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(field, mask, weight)
            loss, l1, l2 = phase_loss(prediction, target, mask, t["l1_weight"], t["l2_weight"])
            excess = cone_loss(prediction * mask, target, cone_cache[size])
            total = loss + t["cone_weight"] * excess
            if not torch.isfinite(total):
                raise FloatingPointError(f"nonfinite loss at step {step}")
            total.backward()
            if t["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            optimizer.step()
            step += 1
            if step % t["log_every"] == 0 or step == steps:
                row = dict(step=step, loss=loss.item(), l1=l1.item(), l2=l2.item(), cone_excess=excess.item(),
                           lr=lr, seconds=time.perf_counter() - started)
                _append(output / "history.csv", fields, row)
                print(" ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                      flush=True)
            if step % t["eval_every"] == 0 or step == steps:
                best, _ = evaluate(model, source, full_cone, output, step, best, config)
            interrupted = stop_after is not None and step >= stop_after
            if step % t["save_every"] == 0 or step == steps or interrupted:
                _save_last(last, model, optimizer, step, best, rng, generator, config)
            if interrupted and step < steps:
                return {"steps": step, "interrupted": True}
        torch.save({"model": model.state_dict(), "config": config, "input_scale": source.scale,
                    "voxel_size": source.voxel_size}, output / "final.pt")
        summary = {"steps": step, "best_nrmse_real": best, "seconds": time.perf_counter() - started}
        if device.type == "cuda":
            summary["peak_gpu_memory_bytes"] = torch.cuda.max_memory_allocated(device)
        (output / "DONE").write_text(json.dumps(summary, indent=2) + "\n")
        return summary
    except BaseException:
        (output / "FAILED").write_text(traceback.format_exc())
        raise


def cli(args):
    with args.config.open("rb") as stream:
        config = tomllib.load(stream)
    for name in ("steps", "output", "device"):
        value = getattr(args, name)
        if value is not None:
            config["training"][name] = str(value) if isinstance(value, Path) else value
    print(json.dumps(run(config, resume=args.resume), indent=2), flush=True)
