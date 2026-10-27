"""The tests of the Deep Fourier-Space Inversion paper on our side (CLAUDE.md 9bo): forward simulations from COSMOS (RC1) at SNR 40, SNR 100,
and SNR 100 with a STRONG PHASE INCONSISTENCY whose phase wraps (+-pi).

  paper_snr40        mask * D*chi_cosmos, notebook complex noise at SNR 40
  paper_snr100       the same at SNR 100
  paper_snr100_wrap  SNR 100 plus a smooth non-dipolar phase psi (random Gaussian field, correlation `--psi-sigma` voxels, rms `--psi-rms` ppm,
                     default 0.1 ppm = 1.9 rad), added BEFORE the complex noise, so the measured phase is wrapped. The target is still
                     mask * D*chi_cosmos (the dipolar part), so eNDI is scored against the same chi_cosmos. The amplitude is our choice (the paper
                     gives none in the abstract). Calibration on RC1: the clean field has rms 0.0084 ppm and wraps only beyond +-0.162 ppm, so any
                     real wrapping needs a non-dipolar phase about 10x the signal: 0.08 ppm wraps 5% of the brain, 0.1 ppm 12%, 0.15 ppm 29%.
                     The wrapped fraction is stored with the test.

Same file layout as export_generalization.py (tag `rc1`, <tag>_common.npz, <tag>_<exp>.npz with `weight` and `field_<method>`), so
`scripts/rescore_fixed.py --dir <out> --sets rc1` scores the eNDI side note at fixed iterations. This script reports the PHASE NRMSE of the
cleaner (percent of the clean field's norm in the brain mask), plus the wrap-invariant one: wrapped differences angle(exp(i s (est - target)))/s,
because exact 2*pi wraps are invisible to eNDI (it uses sin) and a cleaner trained on unwrapped fields is not expected to unwrap.

Run (server): MCNO_VRAM_GB=10 .venv/bin/python scripts/export_paper_tests.py --runs noise_clean paper_e1_rc1 paper_e2_geo paper_e3_ft \
    --out runs/generalization_paper_tests > runs/paper_tests.log 2>&1
Smoke test (laptop, untrained nets): python -I scripts/export_paper_tests.py --smoke --crop 64 --out <empty dir>
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_generalization import clean_with, crop_center, load_model  # noqa: E402

from mcno_qsm.rc1 import load_rc1  # noqa: E402
from mcno_qsm.scoring import forward_field, limit_vram, masked_nrmse  # noqa: E402


def noisy_phase(field, magnitude, snr, phs_scale, rng):
    """Notebook complex-noise recipe: angle(mag exp(i s b) + (randn + i randn) / snr) / s, in ppm (wrapped beyond +-pi / s)."""
    noise = (rng.standard_normal(field.shape) + 1j * rng.standard_normal(field.shape)) / snr
    return (np.angle(magnitude * np.exp(1j * field * phs_scale) + noise) / phs_scale).astype(np.float32)


def smooth_psi(shape, mask, rms, sigma, rng):
    """Random Gaussian-smoothed field with the given rms (ppm) inside the mask."""
    psi = gaussian_filter(rng.standard_normal(shape), sigma)
    return (psi * (rms / psi[mask > 0].std())).astype(np.float32)


def wrap_difference(estimate, target, phs_scale):
    """(estimate - target) wrapped to +-pi / phs_scale: the part of the error that is not a 2*pi multiple."""
    return np.angle(np.exp(1j * phs_scale * (estimate - target))) / phs_scale


def paper_fields(data, magnitude="mag_base", psi_rms=0.1, psi_sigma=12.0, seed=47):
    """{name: (field_ppm, weight, info)} for the three tests. `magnitude`: 'mag_base' (the notebook's, mean ~0.4) or 'magn' (RC1's, mean 0.26)."""
    chi, mask, phs_scale = data["chi"], data["mask"].astype(np.float32), data["phs_scale"]
    clean = forward_field(chi, None, data["voxel_size"])
    if magnitude == "mag_base":
        weight = chi - chi.min()
        weight = (weight / weight.max() * mask).astype(np.float32)
    else:
        weight = data["magnitude"].astype(np.float32)
    rng = np.random.default_rng(seed)
    psi = smooth_psi(chi.shape, mask, psi_rms, psi_sigma, rng)
    inside = mask > 0
    wrapped = float((np.abs(phs_scale * (clean + psi))[inside] > np.pi).mean())
    return {
        "paper_snr40": (noisy_phase(clean, weight, 40, phs_scale, rng), weight, {"snr": 40, "wrapped_fraction": float((np.abs(phs_scale * clean)[inside] > np.pi).mean())}),
        "paper_snr100": (noisy_phase(clean, weight, 100, phs_scale, rng), weight, {"snr": 100, "wrapped_fraction": float((np.abs(phs_scale * clean)[inside] > np.pi).mean())}),
        "paper_snr100_wrap": (noisy_phase(clean + psi, weight, 100, phs_scale, rng), weight,
                              {"snr": 100, "psi_rms_ppm": psi_rms, "psi_sigma_voxels": psi_sigma, "wrapped_fraction": wrapped}),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["noise_clean"], help="run names (runs/s4b_<name>/best.pt)")
    parser.add_argument("--rc1", default="dataRC1")
    parser.add_argument("--out", default="runs/generalization_paper_tests")
    parser.add_argument("--magnitude", choices=["mag_base", "magn"], default="mag_base")
    parser.add_argument("--psi-rms", type=float, default=0.1)
    parser.add_argument("--psi-sigma", type=float, default=12.0)
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
    mask, chi, voxel, phs_scale = data["mask"].astype(np.float32), data["chi"], data["voxel_size"], data["phs_scale"]
    clean = forward_field(chi, None, voxel)
    target = clean * mask
    np.savez_compressed(out / "rc1_common.npz", chi_gt=chi.astype(np.float16), mask=mask.astype(np.uint8), clean=clean.astype(np.float16),
                        weight=data["magnitude"].astype(np.float16), voxel_size=np.array(voxel))
    models = [(name, *load_model(name, device, args.smoke)[:1]) for name in args.runs]
    tests = paper_fields(data, args.magnitude, args.psi_rms, args.psi_sigma)
    norm = float(np.linalg.norm(target[mask > 0]))
    table = {}
    for name, (field, weight, info) in tests.items():
        methods = {"dirty": field * mask}
        for run, model in models:
            methods[run] = clean_with(model, field, mask, weight, clean, voxel, "clean", device)
        np.savez_compressed(out / f"rc1_{name}.npz", weight=weight.astype(np.float16),
                            **{f"field_{m}": v.astype(np.float16) for m, v in methods.items()})
        table[name] = {"info": info, "methods": {}}
        for method, value in methods.items():
            wrapped_err = wrap_difference(value, target, phs_scale)
            table[name]["methods"][method] = {
                "phase_nrmse": masked_nrmse(value, target, mask),
                "phase_nrmse_wrap_invariant": 100.0 * float(np.linalg.norm(wrapped_err[mask > 0])) / norm}
            m = table[name]["methods"][method]
            print(f"{name:18s} {method:16s} phase NRMSE {m['phase_nrmse']:7.2f}%   wrap-invariant {m['phase_nrmse_wrap_invariant']:7.2f}%"
                  f"   (wrapped voxels in the input: {100 * info['wrapped_fraction']:.1f}%)", flush=True)
    (out / "paper_tests.json").write_text(json.dumps({"args": vars(args), "tests": table}, indent=2) + "\n")
    print(f"wrote {out}/paper_tests.json; eNDI side note: scripts/rescore_fixed.py --dir {out} --sets rc1 --methods dirty {' '.join(args.runs)}")


if __name__ == "__main__":
    main()
