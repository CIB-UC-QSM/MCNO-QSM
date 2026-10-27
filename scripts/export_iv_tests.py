"""Test fields for the intravoxel / acquisition-noise runs (CLAUDE.md 9bp), forward simulated on the RC1 head with the weight channel of RC1.

  --set noise   same chi, same target (mask * D*chi_cosmos), SNR 40, other noise statistics: iv_colored40, iv_heavy40, iv_hetero40 (single-echo notebook
                recipe) and iv_acq40 (the multi-echo model with iid Gaussian noise, no sources)
  --set src     three intravoxel sources (|chi| 1.5-2.5 ppm) on the RC1 head, multi-echo model at SNR 100 and 40 (iv_src100, iv_src40; the SAME sources
                in both); the target and chi_gt contain the block-averaged sources, the weight channel carries the dip

Fields are drawn from fixed seeds (not the training seeds, not the monitoring seed). Layout as export_paper_tests.py (tag `rc1`, rc1_common.npz,
rc1_<exp>.npz with `weight` and `field_<method>`), so scripts/rescore_fixed.py --dir <out> --sets rc1 scores the eNDI side. This script prints the
PHASE NRMSE of every method (percent of the target norm in the mask), and for `src` also near the sources (within 4 voxels of a source voxel) and elsewhere.

Run (server): MCNO_VRAM_GB=10 .venv/bin/python scripts/export_iv_tests.py --set src --runs noise_clean iv_noisestat iv_full --out runs/generalization_iv_src
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_generalization import clean_with, crop_center, load_model  # noqa: E402

from mcno_qsm.artifacts import ArtifactConfig, Volume, acquisition, noisy  # noqa: E402
from mcno_qsm.rc1 import load_rc1  # noqa: E402
from mcno_qsm.scoring import limit_vram, masked_nrmse  # noqa: E402

SPECS = {"iv_colored40": {"kind": "colored", "sigma": 1.0}, "iv_heavy40": {"kind": "heavy", "df": 3}, "iv_hetero40": {"kind": "hetero", "depth": 0.5}}


def noise_tests(v, seed):
    out = {}
    for i, (name, spec) in enumerate(SPECS.items()):
        generator = torch.Generator(device=v.clean.device).manual_seed(seed + i)
        out[name] = (noisy(v, v.clean, 40.0, ArtifactConfig(), generator, spec), v.magnitude, v.clean, None, {"snr": 40, "noise": spec["kind"]})
    cfg = ArtifactConfig(snr_low=40.0, snr_high=40.0)
    field, info = acquisition(v, np.random.default_rng(seed), torch.Generator(device=v.clean.device).manual_seed(seed + 9), cfg, sources=False)
    out["iv_acq40"] = (field, v.magnitude, info["clean"], None, {"snr": 40, "noise": "gaussian", "echoes": "4"})
    return out


def source_tests(v, seed):
    out = {}
    for snr in (100.0, 40.0):
        cfg = ArtifactConfig(snr_low=snr, snr_high=snr, iv_count=(3, 3), iv_ppm=(1.5, 2.5))
        field, info = acquisition(v, np.random.default_rng(seed), torch.Generator(device=v.clean.device).manual_seed(seed + int(snr)), cfg)
        out[f"iv_src{int(snr)}"] = (field, info["magnitude"], info["clean"], info["chi_source"], {"snr": snr, "sources": info["sources"]})
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=["noise", "src"], required=True)
    parser.add_argument("--runs", nargs="+", default=["noise_clean"], help="run names (runs/s4b_<name>/best.pt)")
    parser.add_argument("--rc1", default="dataRC1")
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--crop", type=int, default=0, help="smoke test: centre-crop the volume to this size")
    parser.add_argument("--smoke", action="store_true", help="untrained nets (no checkpoints needed)")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        limit_vram(device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data = load_rc1(args.rc1)
    if args.crop:
        data = {**data, **crop_center({k: v for k, v in data.items() if isinstance(v, np.ndarray)}, args.crop)}
    v = Volume.from_rc1(data, device)
    tests = (noise_tests if args.set == "noise" else source_tests)(v, args.seed)
    mask = data["mask"].astype(np.float32)
    first = next(iter(tests.values()))
    chi_source = first[3]
    for name, (_, _, _, chi_s, _) in tests.items():
        if chi_s is not None and not torch.equal(chi_s, chi_source):
            raise RuntimeError("the source tests must share their sources")
    chi_gt = data["chi"] + (0 if chi_source is None else chi_source.cpu().numpy())
    target_field = first[2].cpu().numpy()
    np.savez_compressed(out / "rc1_common.npz", chi_gt=chi_gt.astype(np.float16), mask=mask.astype(np.uint8), clean=target_field.astype(np.float16),
                        weight=data["magnitude"].astype(np.float16), voxel_size=np.array(data["voxel_size"]))
    near = np.zeros_like(mask, bool)
    if chi_source is not None:
        near = ndimage.binary_dilation(np.abs(chi_source.cpu().numpy()) > 0.05, iterations=4)
    models = [(name, load_model(name, device, args.smoke)[0]) for name in args.runs]
    target = target_field * mask
    table = {}
    for name, (field, weight, clean, _, info) in tests.items():
        field, weight = field.cpu().numpy(), weight.cpu().numpy()
        methods = {"dirty": field * mask}
        for run, model in models:
            methods[run] = clean_with(model, field, mask, weight, clean.cpu().numpy(), data["voxel_size"], "clean", device)
        np.savez_compressed(out / f"rc1_{name}.npz", weight=weight.astype(np.float16), **{f"field_{m}": x.astype(np.float16) for m, x in methods.items()})
        table[name] = {"info": info, "methods": {}}
        for method, value in methods.items():
            row = {"phase_nrmse": masked_nrmse(value, target, mask)}
            if near.any():
                row["near_source"] = masked_nrmse(value, target, mask * near)
                row["elsewhere"] = masked_nrmse(value, target, mask * ~near)
            table[name]["methods"][method] = row
            print(f"{name:14s} {method:16s} " + "  ".join(f"{k} {x:7.2f}%" for k, x in row.items()), flush=True)
    (out / "iv_tests.json").write_text(json.dumps({"args": vars(args), "tests": table}, indent=2) + "\n")
    print(f"wrote {out}/iv_tests.json; eNDI: scripts/rescore_fixed.py --dir {out} --sets rc1 --methods dirty {' '.join(args.runs)}")


if __name__ == "__main__":
    main()
