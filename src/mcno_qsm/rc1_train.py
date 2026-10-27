"""Train the PhaseCleaner on RC1 + simulated artifacts (S4b); resumable.

Sim2 (RC2) is the untouched target: no Sim2 voxel is trained on and none is used to choose a checkpoint.
It is evaluated every `eval_every` steps as a monitoring curve only. The checkpoint is selected on the held-out RC1
slab (mean, over the families trained on, of cleaned/dirty phase NRMSE on fixed draws).
"""

import json
import platform
import random
import time
import tomllib
import traceback
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from .artifacts import EXTRA_FAMILIES, FAMILIES, IDENTITY, ArtifactConfig, ObjectConfig
from .cleaner import PhaseCleaner
from .cleaner_train import _append, _file_hash, _learning_rate, _save_last
from .dipolelets import DipoleletTransform
from .losses import band_loss, band_masks, cone_loss, cone_mask, phase_loss, segment_loss
from .metrics import masked_nrmse
from .rc1_data import RC1Source
from .rc2 import load_rc2
from .scoring import forward_field, limit_vram


class Sim2Monitor:
    """Full Sim2 volume for monitoring only (never used for training or selection)."""

    def __init__(self, root, field_strength, device):
        data = load_rc2(root, field_strength)
        t = lambda x: torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)[None, None]
        self.voxel_size, self.affine = data["voxel_size"], data["affine"]
        self.mask, self.weight, self.field = t(data["mask"]), t(data["weight"]), t(data["field"])
        self.target = t(forward_field(data["chi"], None, self.voxel_size)) * self.mask
        self.root = Path(root)

    @torch.no_grad()
    def clean(self, model):
        return model(self.field, self.mask, self.weight, voxel_size=self.voxel_size)

    @torch.no_grad()
    def scores(self, model):
        prediction = self.clean(model)
        return {"sim2_nrmse": masked_nrmse(prediction, self.target, self.mask).item(),
                "sim2_nrmse_input": masked_nrmse(self.field * self.mask, self.target, self.mask).item()}

    def save(self, model, path):
        nib.save(nib.Nifti1Image(self.clean(model).cpu().numpy()[0, 0], self.affine), path)


def build_model(config, source, device):
    """PhaseCleaner by default; `[model] kind = "gain_cascade"` builds the Dipole-let gain network in front of the frozen noise network (CLAUDE.md 9az)."""
    if config["model"].get("kind") == "gain_cascade":
        from .gainnet import GainCascade
        return GainCascade.from_config(source.voxel_size, source.scale, config["dipolelet"], config["model"], device).to(device)
    return PhaseCleaner(source.voxel_size, input_scale=source.scale, dipolelet=config["dipolelet"], **config["model"]).to(device)


def build_source(config, device):
    data, art = config["data"], config.get("artifacts", {})
    cfg = ArtifactConfig(**{k: v for k, v in art.items() if k != "weights"})
    if "weights" in art:
        cfg.weights = {f: float(art["weights"].get(f, 0.0)) for f in (*FAMILIES, *EXTRA_FAMILIES)}
    objects = ObjectConfig(**config.get("objects", {}))
    if "geometric" in data:  # paper experiment 2: brain-free spheres and cylinders; RC1 only for monitoring and selection (CLAUDE.md 9bo)
        from .rc1_data import GeometricSource
        return GeometricSource(data["rc1"], device, cfg, data.get("holdout"), objects=objects, geometric=data["geometric"])
    if "ahead" in data:      # S4c: RC1 + AHEAD heads (CLAUDE.md 9ab)
        from .ahead_data import MixedSource
        return MixedSource(data["rc1"], device, cfg, data.get("holdout"), objects=objects, ahead=data["ahead"])
    return RC1Source(data["rc1"], device, cfg, data.get("holdout"), objects=objects)


@torch.no_grad()
def evaluate(model, source, monitor, output, step, best, config):
    """Fixed-draw scores per family on the RC1 regions, selection ratio, and the Sim2 monitoring row."""
    model.eval()
    tag = "_test" if source.holdout else ""
    row = {"step": step}
    ratios = {}
    for family in (*FAMILIES, *[f for f in EXTRA_FAMILIES if f in source.families]):
        field, info = source.fixed_dirty(family, config.get("eval", {}).get("seed", 12345))
        target, weight = source.target, source.weight     # a family that adds a source brings its own clean field and magnitude
        if "clean" in info:
            target, weight = source.mask * info["clean"][None, None], info["magnitude"][None, None]
        prediction = model(field, source.mask, weight, voxel_size=source.voxel_size)
        for region_tag, region in source.regions.items():
            row[f"{family}{region_tag}"] = masked_nrmse(prediction, target, region).item()
            row[f"{family}_in{region_tag}"] = masked_nrmse(field * source.mask, target, region).item()
        if row[f"{family}_in{tag}"] > 1e-9:  # nothing to remove in the region: no ratio
            ratios[family] = row[f"{family}{tag}"] / row[f"{family}_in{tag}"]
        elif family == IDENTITY:  # nothing to remove: the damage itself (fraction of the field altered) counts as the ratio
            ratios[family] = row[f"{family}{tag}"]
    row["sel_trained"] = float(np.mean([ratios[f] for f in source.families if f in ratios]))
    row["sel_all"] = float(np.mean(list(ratios.values())))
    if monitor is not None:
        row.update(monitor.scores(model))
    if hasattr(source, "ahead_scores"):   # AHEAD validation subjects: monitoring only, not used for selection
        row.update(source.ahead_scores(model, config.get("eval", {}).get("seed", 12345)))
    improved = row["sel_trained"] < best
    if improved:
        best = row["sel_trained"]
        torch.save({"model": model.state_dict(), "config": config, "input_scale": source.scale,
                    "voxel_size": source.voxel_size, "step": step}, output / "best.pt")
    _append(output / "eval.csv", list(row) + ["best"], {**row, "best": best})
    shown = ["step", "sel_trained", "sel_all", "sim2_nrmse", "sim2_nrmse_input"]
    print("eval " + " ".join(f"{k}={row[k]:.5f}" if isinstance(row[k], float) else f"{k}={row[k]}"
                             for k in shown if k in row), flush=True)
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
        if (output / "eval.csv").exists() or last.exists():
            raise FileExistsError(f"{output} already contains a run; choose a new --output or use --resume")
        output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if t["device"] == "auto" \
        else torch.device(t["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "cuda":
        limit_vram(device)   # MCNO_VRAM_GB: the GPU is shared
    try:
        torch.manual_seed(t["seed"])
        random.seed(t["seed"])
        source = build_source(config, device)
        monitor = Sim2Monitor(config["data"]["rc2"], config["data"]["field_strength"], device) \
            if config["data"].get("rc2") else None
        model = build_model(config, source, device)
        if t.get("init_from") and not resume:   # fine-tuning: weights only (no optimizer, no step); the architecture must match
            model.load_state_dict(torch.load(t["init_from"], map_location=device, weights_only=False)["model"])
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=t["learning_rate"], weight_decay=t["weight_decay"])
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
            data = config["data"]
            meta = {
                "config": config, "shape": source.shape, "voxel_size": source.voxel_size,
                "input_scale": source.scale, "families_trained": source.families,
                "holdout": {"axis": source.axis, "train": source.train_range, "test": source.test_range}
                if source.holdout else None,
                "parameters": sum(p.numel() for p in model.parameters()), "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                "python": platform.python_version(), "torch": torch.__version__,
                "rc1_sha256": {p.name: _file_hash(p) for p in sorted(Path(data["rc1"]).glob("*.mat"))},
                **({"ahead": source.ahead_meta} if hasattr(source, "ahead_meta") else {}),
                **({"init_from": t["init_from"]} if t.get("init_from") else {}),
                "sim2_used_for": "monitoring only (no training voxel, no checkpoint selection)",
                "target": "mask * D*chi_cosmos, correctly aligned dipole kernel, unpadded",
            }
            (output / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        size = t["patch"]
        cone_cache = {}
        band_cache = {}
        band_weight = t.get("band_weight", 0.0)   # band-normalized spectral loss, off unless > 0 (CLAUDE.md 9o-bis)
        segment_weight = t.get("segment_weight", 0.0)   # Dipole-let sensitivity-weighted loss, off unless > 0 (CLAUDE.md 9aj)
        segment_gains = t.get("segment_gains")
        if segment_weight > 0 and not segment_gains:
            raise ValueError("segment_weight > 0 needs [training] segment_gains (one eNDI sensitivity per Dipole-let segment)")
        segment_cache = {}
        started = time.perf_counter()
        steps = t["steps"]
        fields = ["step", "loss", "l1", "l2", "cone_excess", *(["band"] if band_weight > 0 else []),
                  *(["seg"] if segment_weight > 0 else []), "lr", "seconds"]
        model.train()
        while step < steps:
            lr = _learning_rate(step, steps, t["learning_rate"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            field, mask, weight, target, voxel_size = source.next_batch(size, rng, generator, t["batch_size"])
            side = field.shape[-1]   # equals `size` unless a voxel-size augmentation shrinks the patch to fit the resampled head
            if (side, voxel_size) not in cone_cache:
                cone_cache[(side, voxel_size)] = cone_mask((side,) * 3, voxel_size, config["dipolelet"]["delta"],
                                                           device=device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(field, mask, weight, voxel_size=voxel_size)
            loss, l1, l2 = phase_loss(prediction, target, mask, t["l1_weight"], t["l2_weight"])
            excess = cone_loss(prediction * mask, target, cone_cache[(side, voxel_size)])
            total = loss + t["cone_weight"] * excess
            band = None
            if band_weight > 0:
                if (side, voxel_size) not in band_cache:
                    band_cache[(side, voxel_size)] = band_masks((side,) * 3, voxel_size, device=device)
                band, _ = band_loss(prediction, target, mask, band_cache[(side, voxel_size)], t.get("band_floor", 1e-3))
                total = total + band_weight * band
            seg = None
            if segment_weight > 0:
                if (side, voxel_size) not in segment_cache:
                    segment_cache[(side, voxel_size)] = DipoleletTransform((side,) * 3, voxel_size, **config["dipolelet"]).to(device)
                seg = segment_loss(prediction, target, mask, segment_cache[(side, voxel_size)], segment_gains)
                total = total + segment_weight * seg
            if not torch.isfinite(total):
                raise FloatingPointError(f"nonfinite loss at step {step}")
            total.backward()
            if t["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            optimizer.step()
            step += 1
            if step % t["log_every"] == 0 or step == steps:
                row = dict(step=step, loss=loss.item(), l1=l1.item(), l2=l2.item(), cone_excess=excess.item(),
                           **({"band": band.item()} if band is not None else {}),
                           **({"seg": seg.item()} if seg is not None else {}), lr=lr, seconds=time.perf_counter() - started)
                _append(output / "history.csv", fields, row)
                print(" ".join(f"{k}={v:.5f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                      flush=True)
            if step % t["eval_every"] == 0 or step == steps:
                best, _ = evaluate(model, source, monitor, output, step, best, config)
            interrupted = stop_after is not None and step >= stop_after
            if step % t["save_every"] == 0 or step == steps or interrupted:
                _save_last(last, model, optimizer, step, best, rng, generator, config)
            if interrupted and step < steps:
                return {"steps": step, "interrupted": True}
        torch.save({"model": model.state_dict(), "config": config, "input_scale": source.scale,
                    "voxel_size": source.voxel_size}, output / "final.pt")
        summary = {"steps": step, "best_sel_trained": best, "seconds": time.perf_counter() - started}
        if (output / "best.pt").exists():
            best_state = torch.load(output / "best.pt", map_location=device, weights_only=False)
            summary["best_step"] = best_state["step"]
            model.load_state_dict(best_state["model"])
            model.eval()
            if monitor is not None:  # cleaned Sim2 volume of the checkpoint chosen on RC1 only (for scripts/score_run.py)
                monitor.save(model, output / "best_clean.nii.gz")
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
