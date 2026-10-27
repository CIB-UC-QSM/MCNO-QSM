"""Gated cascade (CLAUDE.md 9as), server, NO training: stage A = the regime-gated LINEAR Dipole-let filter, stage B = the noise_clean network.

Stage A (the untrained RegimeFilter of 9ar, g0 = 1): x_A = mask * (x - r * sum_s (1 - g1_s) comp_s), with the 12-segment Sim2 Dipole-let
decomposition, g1 = the gains fitted ONLY on RC1's modelled STI / off-diagonal residuals with noise (never on the measured tissue residual,
runs/streak_kspace/regime_init.txt) and r = the local regime map (sigmoid(6 * (log cone excess of the third detail scale - log white-noise ratio)
+ 9.0848), r = 0.5 at excess 0.22). Stage B = runs/s4b_noise_clean/best.pt applied to x_A. Methods per experiment (eNDI, ndiCG, alpha 1e-5,
40 iterations stored, read at 25):
  dirty | B (noise_clean alone, reproduces the table) | A (stage A alone) | AB (A then B, soft gate) | ABh (A then B, hard gate r > 0.5)
Experiments: RC1 exp1/3/5/6/7/8/9 and the forward Sim2 sets from the export directory, the REAL Sim2, and two uncalibrated STRESS tests of background-field
remnants (bg0.3, bg1.0: RC1 noise-SNR-60 field + the masked dipole field of smooth random susceptibility OUTSIDE the brain mask, scaled to 0.3 / 1.0 x the
rms of the clean field). The remnant tests have no calibration to in-vivo data; they only show how each stage reacts to a smooth non-dipolar-looking field.
Optional `--gain-run NAME` (CLAUDE.md 9az) adds two columns: GNa = the trained gain network (stage A') alone, GN = A' then the frozen noise_clean inside it
(runs/s4b_NAME/best.pt). The other columns are unchanged.
Optional `--norm-run NAME` (CLAUDE.md 9bd) adds the column Bn = runs/s4b_NAME/best.pt (a PhaseCleaner, e.g. noise_clean_norm) applied alone, like B.
Run (server): MCNO_VRAM_GB=10 nohup .venv/bin/python scripts/cascade_gate.py > runs/cascade.log 2>&1 &    Dry run (laptop): python -I scripts/cascade_gate.py --dry
Writes runs/cascade/fixed_meta.json (resumable, keys tag/exp/method like rescore_fixed.py) and table.txt."""

import argparse
import json
import sys
import tomllib
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rescore_fixed import invert, line  # noqa: E402

from mcno_qsm.artifacts import ArtifactConfig, Volume, make_dirty  # noqa: E402
from mcno_qsm.cleaner import PhaseCleaner, gaussian_smooth  # noqa: E402
from mcno_qsm.dipolelets import SIM2_DIPOLELET  # noqa: E402
from mcno_qsm.gainnet import GainCascade  # noqa: E402
from mcno_qsm.rc1 import clean_field, load_rc1  # noqa: E402
from mcno_qsm.rc2 import load_rc2  # noqa: E402
from mcno_qsm.scoring import forward_field, limit_vram  # noqa: E402

DIP = {**SIM2_DIPOLELET, "coarse_bands": 3}
# runs/streak_kspace/regime_init.txt (fit_regime_init.py): segments scale1..3/band0..2, coarse/band0..2
G1 = [0.4829, -0.3861, -0.0269, -1.2506, 1.1527, 1.2623, -0.1013, 0.5851, 0.7612, -1.413, 0.8481, 0.9137]
G1_PASSIVE = [0.0, 0.0, 0.398, 0.0, 0.744, 1.0, 0.0, 0.773, 0.828, 0.0, 0.674, 0.919]    # g1 refitted with bounds [0, 1] (check_passive_filter.py, 9au)
W, C = [0.0, 0.0, 6.0, 0.0], 9.0848
SETTINGS = {"rc1": (25e-3, 2.8936, 192), "sim2": (8e-3, 7.0, 256)}
METHODS = ["dirty", "B", "A", "AB", "ABh"]


def load_noise_clean(device, smoke):
    if smoke:
        config = tomllib.loads(Path("configs/s4b_noise_clean.toml").read_text())
        return PhaseCleaner((1.0, 1.0, 1.0), input_scale=0.0084, dipolelet=config["dipolelet"], **config["model"]).to(device).eval()
    state = torch.load(Path("runs/s4b_noise_clean/best.pt"), map_location=device, weights_only=False)
    model = PhaseCleaner(state["voxel_size"], input_scale=state["input_scale"], dipolelet=state["config"]["dipolelet"], **state["config"]["model"])
    model.load_state_dict(state["model"])
    return model.to(device).eval()


def load_gain_run(name, device, smoke, model_b):
    if smoke:     # laptop dry run: untrained U-Net (identity-of-the-prior at init) in front of the untrained stage B
        config = tomllib.loads(Path("configs/s4b_gainnet.toml").read_text())
        options = {k: v for k, v in config["model"].items() if k not in ("kind", "stage_b")}
        return GainCascade((1.0, 1.0, 1.0), stage_b=model_b, input_scale=0.0084, dipolelet=config["dipolelet"], gain=options).to(device).eval()
    state = torch.load(Path("runs") / f"s4b_{name}" / "best.pt", map_location=device, weights_only=False)
    model = GainCascade.from_config(state["voxel_size"], state["input_scale"], state["config"]["dipolelet"], state["config"]["model"], device)
    model.load_state_dict(state["model"])
    return model.to(device).eval()


@torch.no_grad()
def gain_outputs(model, field, mask, weight, reference, voxel, outside, device):
    """Same input convention as stage_b(): the cascade gets the full field (noise-free D*chi outside the mask for 'clean', zeros for 'out0')."""
    full = field * mask if outside == "out0" else field * mask + reference * (1 - mask)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)[None, None]
    f, m, w = t(full), t(mask), t(weight)
    return model.stage_a(f, m, w, voxel)[0, 0].cpu().numpy(), model(f, m, w, voxel_size=voxel)[0, 0].cpu().numpy()


def load_b_run(name, device, smoke):
    """A PhaseCleaner run (e.g. noise_clean_norm) applied alone; smoke = untrained model of its config on the laptop."""
    if smoke:
        config = tomllib.loads(Path(f"configs/s4b_{name}.toml").read_text())
        return PhaseCleaner((1.0, 1.0, 1.0), input_scale=0.0084, dipolelet=config["dipolelet"], **config["model"]).to(device).eval()
    state = torch.load(Path("runs") / f"s4b_{name}" / "best.pt", map_location=device, weights_only=False)
    model = PhaseCleaner(state["voxel_size"], input_scale=state["input_scale"], dipolelet=state["config"]["dipolelet"], **state["config"]["model"])
    model.load_state_dict(state["model"])
    return model.to(device).eval()


class StageA:
    """`strong = (lo, hi)`: strong-source suppressor (CLAUDE.md 9at). The gate is multiplied by 1 - smoothstep((lrms - lo) / (hi - lo)), lrms = local rms
    (Gaussian sigma 6 voxels) of the input field in ppm; lo / hi were read from RC1 only (real-residual and noise fields stay below 0.032 ppm, the
    strong-source experiments exp5 / exp6 reach 0.09-0.27)."""

    def __init__(self, voxel, device, strong=None, g1=None, strong_sigma=6.0):
        g1 = G1 if g1 is None else list(g1)
        self.device, self.strong, self.strong_sigma = device, strong, float(strong_sigma)
        self.model = PhaseCleaner(voxel, dipolelet=DIP, channels=2, blocks=1, dilations=(1,), checkpoint=False,
                                  regime=dict(rho=6.0, g0=[1.0] * len(g1), g1=g1, w=W, c=C)).to(device).eval()
        self.removal1 = torch.as_tensor([1.0 - g for g in g1], dtype=torch.float32, device=device)

    @torch.no_grad()
    def __call__(self, field, mask, voxel):
        """-> (soft, hard, r): masked stage-A outputs for the soft gate and for r > 0.5, and the regime map."""
        x = torch.from_numpy(np.ascontiguousarray(field * mask, np.float32)).to(self.device)[None, None]
        transform, _, mass = self.model._geometry(x.shape[-3:], x.device, voxel)
        comps = torch.stack([torch.fft.ifftn(s, dim=(-3, -2, -1), norm="ortho").real[:, 0] for s in transform.analysis(x)], 1)
        _, r = self.model.regime(comps, mass)
        if self.strong is not None:
            lo, hi = self.strong
            t = ((gaussian_smooth(x.square(), self.strong_sigma).clamp_min(0).sqrt() - lo) / (hi - lo)).clamp(0, 1)
            r = r * (1 - t * t * (3 - 2 * t))
        full = torch.einsum("s,bsxyz->bxyz", self.removal1, comps)[:, None]
        m = torch.from_numpy(mask).to(self.device)[None, None]
        soft, hard = m * (x - r * full), m * (x - (r > 0.5).float() * full)
        return soft[0, 0].cpu().numpy(), hard[0, 0].cpu().numpy(), r[0, 0].cpu().numpy()


@torch.no_grad()
def stage_b(model, field, mask, weight, reference, voxel, outside, device):
    """Same input convention as export_generalization.clean_with ('clean' = noise-free D*chi outside the mask, 'out0' = zeros)."""
    field = field * mask if outside == "out0" else field * mask + reference * (1 - mask)
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)).to(device)[None, None]
    return model(t(field), t(mask), t(weight), voxel_size=voxel)[0, 0].cpu().numpy()


def remnant(chi_shape, mask, voxel, target_rms, seed):
    """Masked dipole field of smooth random susceptibility OUTSIDE the mask (a crude stand-in for imperfect background removal)."""
    from scipy.ndimage import distance_transform_edt
    rng = np.random.default_rng(seed)
    dist = distance_transform_edt(mask == 0) * np.mean(voxel)
    sources = gaussian_filter(rng.standard_normal(chi_shape).astype(np.float32), 5.0) * ((dist > 2.0) & (dist < 40.0))
    field = forward_field(sources, np.ones(chi_shape, np.float32), voxel) * mask
    return (field * target_rms / np.sqrt(np.mean(field[mask > 0] ** 2))).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("runs/generalization_regime"), help="export directory with <tag>_common.npz and <tag>_<exp>.npz")
    parser.add_argument("--rc1", default="dataRC1")
    parser.add_argument("--rc2", default="dataRC2")
    parser.add_argument("--rc1-exps", nargs="*", default=["exp1", "exp3", "exp5", "exp6", "exp7", "exp8", "exp9"])
    parser.add_argument("--sim2-exps", nargs="*", default=["sim2_snr100", "sim2_snrcal", "sim2_snr40", "sim2_jumps", "sim2_spheres"])
    parser.add_argument("--bg", nargs="*", type=float, default=[0.3, 1.0])
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--passive", action="store_true", help="use the passive g1 (gains bounded to [0, 1], CLAUDE.md 9au)")
    parser.add_argument("--strong", nargs=2, type=float, default=None, metavar=("LO", "HI"), help="strong-source suppressor thresholds in ppm (9at)")
    parser.add_argument("--gain-run", default=None, help="name of a s4b gain_cascade run: adds the columns GNa (A' alone) and GN (A' then B), 9az")
    parser.add_argument("--norm-run", default=None, help="name of a PhaseCleaner run applied alone as the extra column Bn (9bd)")
    parser.add_argument("--out", type=Path, default=Path("runs/cascade"))
    parser.add_argument("--dry", action="store_true", help="laptop smoke test: untrained stage B, 64^3 crop, 2 iterations, rc1 exp7 + real Sim2 + bg0.3 only")
    args = parser.parse_args()
    if args.dry:
        args.iters, args.rc1_exps, args.sim2_exps, args.bg = 2, ["exp7"], [], [0.3]
    limit_vram()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    meta_path = args.out / ("fixed_meta_dry.json" if args.dry else "fixed_meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    crop = (lambda a: a[tuple(slice((n - 64) // 2, (n - 64) // 2 + 64) for n in a.shape[:3])]) if args.dry else (lambda a: a)
    model_b = load_noise_clean(device, args.dry)
    gain_model = load_gain_run(args.gain_run, device, args.dry, model_b) if (args.gain_run or args.dry) else None
    norm_model = load_b_run(args.norm_run, device, args.dry) if args.norm_run else None
    methods = METHODS + (["GNa", "GN"] if gain_model is not None else []) + (["Bn"] if norm_model is not None else [])

    def score(tag, exp, fields, chi, mask, weight, reference, voxel, outside):
        te, b0, pad = SETTINGS[tag]
        stage = StageA(voxel, device, tuple(args.strong) if args.strong else None, G1_PASSIVE if args.passive else None)
        dirty = fields
        soft, hard, r = stage(dirty, mask, voxel)
        out = {"dirty": dirty * mask, "B": stage_b(model_b, dirty, mask, weight, reference, voxel, outside, device), "A": soft}
        out["AB"] = stage_b(model_b, soft, mask, weight, reference, voxel, outside, device)
        out["ABh"] = stage_b(model_b, hard, mask, weight, reference, voxel, outside, device)
        print(f"{tag} {exp}: regime map r in the mask p5/p50/p95 = {np.percentile(r[mask > 0], 5):.2f}/{np.percentile(r[mask > 0], 50):.2f}/"
              f"{np.percentile(r[mask > 0], 95):.2f}, mean {r[mask > 0].mean():.2f}", flush=True)
        if norm_model is not None:
            out["Bn"] = stage_b(norm_model, dirty, mask, weight, reference, voxel, outside, device)
        if gain_model is not None:
            out["GNa"], out["GN"] = gain_outputs(gain_model, dirty, mask, weight, reference, voxel, outside, device)
        for method in methods:
            key = f"{tag}/{exp}/{method}"
            if key not in meta:
                meta[key] = invert(out[method].astype(np.float32), chi, mask, weight, te, b0, pad, args.iters)
                meta[key]["r_mean"] = float(r[mask > 0].mean())
                meta_path.write_text(json.dumps(meta))
                print(line(f"{tag} {exp} {method}", meta[key]), flush=True)

    # ---- RC1 and forward Sim2 sets (fields from the export directory)
    for tag, exps in (("rc1", args.rc1_exps), ("sim2", args.sim2_exps)):
        if not exps or (args.dry and not (args.dir / f"{tag}_common.npz").exists()):    # the exports live on the server
            continue
        c = np.load(args.dir / f"{tag}_common.npz")
        chi, mask, ref, wt = (crop(c[k]).astype(np.float32) for k in ("chi_gt", "mask", "clean", "weight"))
        voxel = tuple(float(v) for v in c["voxel_size"])
        for exp in exps:
            z = np.load(args.dir / f"{tag}_{exp}.npz")
            weight = crop(z["weight"]).astype(np.float32) if "weight" in z.files else wt
            score(tag, exp, crop(z["field_dirty"]).astype(np.float32), chi, mask, weight, ref, voxel, "clean")

    # ---- real Sim2 (zero outside the mask, as in the data)
    d2 = load_rc2(args.rc2, 7.0)
    m2 = d2["mask"].astype(np.float32)
    score("sim2", "sim2_real", crop(d2["field"] * m2), crop(d2["chi"]), crop(m2), crop(d2["weight"]), crop(forward_field(d2["chi"], m2, d2["voxel_size"])),
          d2["voxel_size"], "out0")

    # ---- background-remnant stress tests on RC1 (not calibrated to in-vivo data)
    if args.bg:
        d = load_rc1(args.rc1)
        d = {k: (crop(v) if isinstance(v, np.ndarray) and v.ndim >= 3 else v) for k, v in d.items()}
        mask, chi, voxel = d["mask"].astype(np.float32), d["chi"], d["voxel_size"]
        b = clean_field(chi, voxel, mask)
        noisy, _ = make_dirty("noise", Volume.from_rc1(d, "cpu"), np.random.default_rng(5), torch.Generator().manual_seed(5),
                              ArtifactConfig(snr_low=60.0, snr_high=60.0))
        rms = float(np.sqrt(np.mean(b[mask > 0] ** 2)))
        for amp in args.bg:
            rem = remnant(chi.shape, mask, voxel, amp * rms, seed=7)
            score("rc1", f"bg{amp:g}", noisy.numpy() * mask + rem, chi, mask, d["magnitude"], b, voxel, "clean")

    # ---- table with the references of the regime run
    ref_path = args.dir / "fixed_meta.json"
    refs = json.loads(ref_path.read_text()) if ref_path.exists() else {}
    at = lambda h, i=25: h["nrmse"][min(i, len(h["nrmse"]) - 1)]
    rows = [f"eNDI NRMSE %, 25 fixed iterations (A = gated linear filter, B = noise_clean, AB = A then B, ABh = hard gate; refs = regime run table)",
            f"{'experiment':16s}" + "".join(f"{m:>9s}" for m in methods) + f"{'r_mean':>8s}{'ref nc':>9s}{'ref reg':>9s}"]
    for key in sorted({k.rsplit("/", 1)[0] for k in meta}):
        cells = "".join(f"{at(meta[f'{key}/{m}']):9.2f}" if f"{key}/{m}" in meta else f"{'-':>9s}" for m in methods)
        r_mean = meta.get(f"{key}/dirty", {}).get("r_mean", float("nan"))
        ref = lambda m: f"{at(refs[f'{key}/{m}']):9.2f}" if f"{key}/{m}" in refs else f"{'-':>9s}"
        rows.append(f"{key[key.index('/') + 1:]:16s}{cells}{r_mean:8.2f}{ref('noise_clean')}{ref('regime')}")
    text = "\n".join(rows)
    print(text)
    (args.out / ("table_dry.txt" if args.dry else "table.txt")).write_text(text + "\n")


if __name__ == "__main__":
    main()
