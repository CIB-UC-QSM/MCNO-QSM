"""Generalization export (run on the GPU server): the paper-notebook experiments and forward-simulated Sim2.

Two sets, each cleaned by the trained nets and (optionally) inverted with the user's eNDI:
  rc1   experiments exp1..exp11 of utils/Dipole-lets_Paper_Plots_3.ipynb (cell 6, same positions, seed 47, same
        recipes) on the RC1 head. CAVEAT: the nets were trained on this head (x in [0,100) of axis 0), with random
        placements of the same artifact families, so only the check slab x in [112,160) is held out, and only
        partly (see CLAUDE.md 9f).
  sim2  the Sim2 ground-truth chi forward-simulated (D*chi) with the notebook's complex-noise recipe. Sim2 anatomy was
        never trained on; only the noise is synthetic (so the true psi of Sim2 is NOT in these inputs).
Every field is cleaned under two input conventions for the region outside the brain mask: "clean" (noise-free D*chi
there, as in training) and "out0" (zero there, as Sim2's Frequency.nii and a BET-masked in-vivo field).
One .npz per experiment (float16 volumes) plus a common file and a meta json; existing files are skipped.

Run (server): MCNO_VRAM_GB=10 nohup .venv/bin/python scripts/export_generalization.py --runs noise mix \
    --out runs/generalization > runs/generalization.log 2>&1 &
Smoke test (laptop, untrained nets, no eNDI): python -I scripts/export_generalization.py --smoke --crop 112
"""

import argparse
import json
import sys
import tomllib
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_phantoms import create_geometric_phantom  # noqa: E402

from mcno_qsm.cleaner import PhaseCleaner  # noqa: E402
from mcno_qsm.noise import calibrate_snr, simulate_dirty  # noqa: E402
from mcno_qsm.rc1 import load_rc1, sti_fields  # noqa: E402
from mcno_qsm.rc2 import load_rc2, phase_scale  # noqa: E402
from mcno_qsm.scoring import forward_field, limit_vram, masked_nrmse, score_endi  # noqa: E402

SIM2_PSI_RMS = 0.0037  # ppm, measured rms of Sim2's psi in the mask (CLAUDE.md 9c)


def load_model(name, device, smoke):
    if smoke:
        config = tomllib.loads(Path("configs/s4b_noise.toml").read_text())
        model = PhaseCleaner((1.0, 1.0, 1.0), input_scale=0.0084, dipolelet=config["dipolelet"], **config["model"])
        step = 0
    else:
        state = torch.load(Path("runs") / f"s4b_{name}" / "best.pt", map_location=device, weights_only=False)
        config, step = state["config"], state["step"]
        model = PhaseCleaner(state["voxel_size"], input_scale=state["input_scale"], dipolelet=config["dipolelet"],
                             **config["model"])
        model.load_state_dict(state["model"])
    return model.to(device).eval(), step


@torch.no_grad()
def clean_with(model, field, mask, weight, reference, voxel, outside, device):
    """Cleaned field (numpy). `outside`: 'clean' = noise-free D*chi outside the mask (training), 'out0' = zeros."""
    if outside == "out0":
        field = field * mask
    else:
        field = field * mask + reference * (1 - mask)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)[None, None]
    return model(t(field), t(mask), t(weight), voxel_size=voxel)[0, 0].cpu().numpy()


def crop_center(data, size):
    out = dict(data)
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.ndim >= 3:
            starts = [(n - size) // 2 for n in value.shape[:3]]
            out[key] = value[tuple(slice(s, s + size) for s in starts)]
    return out


def rc1_experiments(data):
    """exp1..exp11 of the paper notebook (cell 6). Returns {name: (field_ppm, description)}."""
    chi, mask = data["chi"], data["mask"].astype(np.float32)
    N, voxel, phs_scale = chi.shape, data["voxel_size"], data["phs_scale"]
    clean = forward_field(chi, None, voxel)
    mag_base = chi - chi.min()
    mag_base = mag_base / mag_base.max() * mask
    psi_star = np.zeros_like(clean)
    psi_star[N[0] // 2, N[1] // 2 + 20, N[2] // 2] = 2
    psi_star[N[0] // 2, N[1] // 2 - 45, N[2] // 2] = 2
    psi_star[N[0] // 2, N[1] // 2 + 40, N[2] // 2 - 20] = -2
    psi_star[N[0] // 2, N[1] // 2 + 40, N[2] // 2 + 25] = -2
    psi_star[N[0] // 2, N[1] // 2, N[2] // 2 + 10] = -1
    psi_star[N[0] // 2, N[1] // 2, N[2] // 2 - 10] = 1

    np.random.seed(47)

    def noisy_phase(field, snr):
        signal = mag_base * np.exp(1j * field * phs_scale) + (1 / snr) * (np.random.randn(*N) + 1j * np.random.randn(*N))
        return (np.angle(signal) / phs_scale).astype(np.float32)

    phase1 = noisy_phase(clean, 100)
    phase2 = noisy_phase(clean, 40)
    sphere = lambda c, v: create_geometric_phantom(matrix_size=N, shape_type="sphere", size=(4, 4, 4), center=c,
                                                   intensity=v)
    cylinder = lambda c, angles, v: create_geometric_phantom(matrix_size=N, shape_type="cylinder", size=(8, 8, 8),
                                                             center=c, rotation_angles=angles, intensity=v)
    s1 = sphere((N[0] // 2 + 10, N[1] // 2 - 35, N[2] // 2), 0.33)
    s2 = sphere((N[0] // 2 - 15, N[1] // 2 + 45, N[2] // 2), -0.33)
    s3 = sphere((N[0] // 2 + 10, N[1] // 2 + 45, N[2] // 2), 0.33)
    c1 = cylinder((N[0] // 2 + 10, N[1] // 2 - 35, N[2] // 2), (45, 45, 45), 0.5)
    c2 = cylinder((N[0] // 2 - 25, N[1] // 2 + 25, N[2] // 2), (30, 30, 30), -0.5)
    artifact = forward_field((chi + s1 + s2 + s3).astype(np.float32), None, voxel)
    phase3 = noisy_phase(artifact, 40)       # the notebook draws this one with the SNR-40 level (noise2)
    pSTI, pQ = sti_fields(data["chi_13"], data["chi_23"], data["chi_33"], voxel)
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    return {
        "exp1": (clean, "noiseless"),
        # noise drawn with the notebook's magnitude (mag_base), so the weight channel must be mag_base too (noise level cue)
        "exp2": (phase1, "noise SNR 100", f32(mag_base)),
        "exp3": (phase2, "noise SNR 40", f32(mag_base)),
        "exp4": (f32(clean + psi_star), "phase jumps"),
        "exp5": (f32(clean + c1 + c2), "cylinders"),
        "exp6": (f32(clean + s1 + s2 + s3), "spheres (phase)"),
        "exp7": (data["phs_tissue"], "in vivo phs_tissue"),
        "exp8": (f32(pSTI), "STI"),
        "exp9": (f32(clean - pQ), "off-diagonal (clean - pQ)"),
        "exp10": (clean, "masking only (mask applied by the net)"),
        "exp11": (phase3, "spheres in chi + noise SNR 40", f32(mag_base)),
    }


def sim2_experiments(data, device):
    """Forward-simulated Sim2: chi_gt -> D*chi -> notebook complex-noise recipe (magnitude = the raw eNDI weight)."""
    chi, mask, weight, voxel = data["chi"], data["mask"].astype(np.float32), data["weight"], data["voxel_size"]
    N = chi.shape
    phs = phase_scale(8e-3, 7.0)
    clean = forward_field(chi, None, voxel)
    # Keep the RAW eNDI weight as magnitude: its in-mask mean (0.25) matches RC1 `magn` (0.26), the weight channel the nets were
    # trained with. Rescaling it to 0..1 (mean 0.71) puts the net far outside its training range (measured: phase error > dirty).
    magnitude = (weight * mask).astype(np.float32)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)
    clean_t, mag_t, mask_t = t(clean), t(magnitude), t(mask)
    snr_cal = calibrate_snr(clean_t, mag_t, mask_t, phs, SIM2_PSI_RMS)
    print(f"Sim2 forward simulation: SNR matching psi rms {SIM2_PSI_RMS} ppm = {snr_cal:.1f}", flush=True)
    generator = torch.Generator(device=device).manual_seed(47)

    def noisy(field, snr):
        return simulate_dirty(t(field), mag_t, snr, phs, generator).cpu().numpy()

    c = [int(v) for v in np.argwhere(mask > 0).mean(0)]
    jumps = np.zeros_like(clean)
    for offset, value in [((0, 20, 0), 2), ((0, -30, 0), 2), ((0, 30, -20), -2), ((0, 30, 25), -2), ((0, 0, 10), -1),
                          ((0, 0, -10), 1)]:
        jumps[tuple(ci + oi for ci, oi in zip(c, offset))] = value
    spheres = sum(create_geometric_phantom(matrix_size=N, shape_type="sphere", size=(4, 4, 4),
                                           center=tuple(ci + oi for ci, oi in zip(c, offset)), intensity=v)
                  for offset, v in [((10, -35, 0), 0.33), ((-15, 45, 0), -0.33), ((10, 45, 0), 0.33)])
    experiments = {
        "sim2_snr100": (noisy(clean, 100), "forward-simulated, SNR 100"),
        "sim2_snr40": (noisy(clean, 40), "forward-simulated, SNR 40"),
        "sim2_snrcal": (noisy(clean, snr_cal), f"forward-simulated, SNR {snr_cal:.0f} (psi rms of Sim2)"),
        "sim2_jumps": (noisy(clean, snr_cal) + jumps.astype(np.float32), "SNR-cal noise + phase jumps"),
        "sim2_spheres": (noisy(forward_field((chi + spheres).astype(np.float32), None, voxel), snr_cal),
                         "SNR-cal noise + spheres in chi"),
    }
    # the noise was drawn with `magnitude`, so that (not the raw eNDI weight) is the noise-level channel of the net
    return {k: (*v, magnitude) for k, v in experiments.items()}, clean, magnitude


def run_set(tag, experiments, ctx, models, args, device):
    out = Path(args.out)
    mask, chi, weight, voxel = ctx["mask"], ctx["chi"], ctx["weight"], ctx["voxel"]
    clean_full = ctx["clean"]
    target = clean_full * mask
    n0 = mask.shape[0]
    regions = {"all": mask}
    if tag == "rc1":
        check = np.zeros_like(mask)
        check[int(round(0.7 * n0)):] = 1
        regions["check"] = mask * check
        train = np.zeros_like(mask)
        train[: int(round(0.625 * n0))] = 1
        regions["train"] = mask * train
    common = out / f"{tag}_common.npz"
    if not common.exists() or args.force:
        np.savez_compressed(common, chi_gt=chi.astype(np.float16), mask=mask.astype(np.uint8), clean=clean_full.astype(np.float16),
                            weight=weight.astype(np.float16), voxel_size=np.array(voxel))
    meta_path = out / f"{tag}_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    score_kw = dict(te=ctx["te"], b0=ctx["b0"], pad=ctx["pad"])
    if args.endi != "none" and "oracle" not in meta:
        r = score_endi(target, chi, mask, weight, **score_kw)
        meta["oracle"] = {k: r[k] for k in ("nrmse", "xsim", "iterations", "nrmse_history", "xsim_history")}
        np.savez_compressed(out / f"{tag}_oracle.npz", chi=r["chi"].astype(np.float16))
        print(f"[{tag}] oracle eNDI {r['nrmse']:.2f}% XSIM {r['xsim']:.4f} stop {r['iterations']}", flush=True)
        meta_path.write_text(json.dumps(meta))
    for name, spec in experiments.items():
        field, description = spec[0], spec[1]
        w = spec[2] if len(spec) > 2 else weight          # per-experiment weight / noise-level channel
        if args.only and name not in args.only:
            continue
        file = out / f"{tag}_{name}.npz"
        if file.exists() and not args.force:
            print(f"[{tag}] skip {name} (exists)", flush=True)
            continue
        methods = {"dirty": field * mask}
        for run, model in models:
            for outside, suffix in (("clean", ""), ("out0", "_out0")):
                methods[run + suffix] = clean_with(model, field, mask, w, clean_full, voxel, outside, device)
        if args.endi == "all":
            scored = list(methods)
        elif args.endi == "main":
            scored = ["dirty", models[0][0], models[0][0] + "_out0"]
        else:
            scored = []
        row = {"description": description, "methods": {}}
        arrays = {} if len(spec) < 3 else {"weight": w.astype(np.float16)}
        for method, value in methods.items():
            entry = {f"phase_nrmse_{k}": masked_nrmse(value, target, r) for k, r in regions.items() if r.any()}
            arrays[f"field_{method}"] = value.astype(np.float16)
            if method in scored:
                try:
                    r = score_endi(value, chi, mask, w, **score_kw)
                    entry.update({"endi_nrmse": r["nrmse"], "xsim": r["xsim"], "stop": r["iterations"],
                                  "nrmse_history": r["nrmse_history"], "xsim_history": r["xsim_history"]})
                    arrays[f"chi_{method}"] = r["chi"].astype(np.float16)
                except Exception:   # a failed inversion must not lose the experiment
                    entry["endi_error"] = traceback.format_exc(limit=2)
            row["methods"][method] = entry
            print(f"[{tag}] {name:12s} {method:12s} phase {entry['phase_nrmse_all']:6.2f}%"
                  + (f"  eNDI {entry['endi_nrmse']:6.2f}% XSIM {entry['xsim']:.4f} stop {entry['stop']}"
                     if "endi_nrmse" in entry else ""), flush=True)
        np.savez_compressed(file, **arrays)
        meta[name] = row
        meta_path.write_text(json.dumps(meta))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["noise", "mix"])
    parser.add_argument("--rc1", default="dataRC1")
    parser.add_argument("--rc2", default="dataRC2")
    parser.add_argument("--out", default="runs/generalization")
    parser.add_argument("--sets", nargs="+", default=["rc1", "sim2"])
    parser.add_argument("--endi", choices=["none", "main", "all"], default="main",
                        help="main: eNDI of dirty and of the first run (both conventions); all: of every field")
    parser.add_argument("--pad-rc1", type=int, default=192, help="eNDI padding for RC1 (160^3); Sim2 always uses 256")
    parser.add_argument("--crop", type=int, default=0, help="smoke test: centre-crop the volumes to this size")
    parser.add_argument("--smoke", action="store_true", help="untrained nets (no checkpoints needed)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", nargs="+", default=None, help="process only these experiment names")
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        limit_vram(device)
    models = []
    for name in args.runs:
        model, step = load_model(name, device, args.smoke)
        models.append((name, model))
        print(f"loaded {name} (best step {step})", flush=True)

    if "rc1" in args.sets:
        data = load_rc1(args.rc1)
        if args.crop:
            data = crop_center(data, args.crop)
        ctx = {"chi": data["chi"], "mask": data["mask"].astype(np.float32), "weight": data["magnitude"],
               "voxel": data["voxel_size"], "clean": forward_field(data["chi"], None, data["voxel_size"]),
               "te": data["te"], "b0": data["b0"], "pad": args.pad_rc1}
        run_set("rc1", rc1_experiments(data), ctx, models, args, device)
    if "sim2" in args.sets:
        data = load_rc2(args.rc2, 7.0)
        if args.crop:
            data = crop_center(data, args.crop)
        experiments, clean, _ = sim2_experiments(data, device)
        ctx = {"chi": data["chi"], "mask": data["mask"].astype(np.float32), "weight": data["weight"],
               "voxel": data["voxel_size"], "clean": clean, "te": 8e-3, "b0": 7.0, "pad": 256}
        run_set("sim2", experiments, ctx, models, args, device)
    print("export finished", flush=True)


if __name__ == "__main__":
    main()
