"""Plug-and-play residual refinement inside eNDI (R2, CLAUDE.md 9aw), server, NO training.

Round 0: b_0 = C(phi) (the one-pass cascade, same input convention as cascade_gate.py), x = eNDI(b_0), n0 + 1 steps from zero.
Rounds 1..R: r = mask * (phi - K x) (the residual of the current estimate, ppm; K = the padded eNDI kernel), b = mask * K x + C(r), x = eNDI(b) warm-started
from x (`precond` of ndiCG, user's module untouched), n + 1 steps. Total steps 26 = the 25 fixed iterations of every other table (history index 25).
C = cascade operator, variants: AB (stage A passive, then noise_clean B; PRIMARY), B (noise_clean alone), A (stage A alone, diagnostic).
Residual inputs to B use the zeros-outside convention. Reference rows (AB1 = the one-pass cascade, recomputed here) and the criteria are in CLAUDE.md 9aw.
Run (server): MCNO_VRAM_GB=10 nohup .venv/bin/python scripts/pnp_endi.py > runs/pnp.log 2>&1 &     Dry run (laptop): python -I scripts/pnp_endi.py --dry --dir runs/generalization
Writes runs/pnp/fixed_meta.json (resumable) and table.txt."""

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cascade_gate import G1_PASSIVE, SETTINGS, StageA, load_noise_clean, stage_b  # noqa: E402

from mcno_qsm.rc2 import load_rc2, phase_scale  # noqa: E402
from mcno_qsm.scoring import _modules, _pad_corner, forward_field, limit_vram  # noqa: E402

VARIANTS = ["AB", "B", "A"]


class Solver:
    """ndiCG wrapper with warm start; keeps the padded estimate (radians) between rounds."""

    def __init__(self, tag, chi, mask, weight, device):
        te, b0, self.pad = SETTINGS[tag]
        self.tools, self.endi = _modules()
        self.scale = phase_scale(te, b0)
        self.device, self.shape = device, mask.shape
        self.crop = tuple(slice(0, n) for n in mask.shape)
        self.kernel = self.tools.dipole_kernel((self.pad,) * 3, (self.pad,) * 3)
        self.kernel_t = torch.as_tensor(np.asarray(self.kernel), dtype=torch.float32, device=device)
        self.chi, self.mask, self.weight = chi, mask, weight

    def solve(self, field, x0, n):
        params = {"K": self.kernel, "input": _pad_corner(field * self.scale, self.pad), "alpha": 1e-5, "isNonlinear": False,
                  "weight": _pad_corner(self.weight, self.pad), "mask": _pad_corner(self.mask, self.pad), "maxOuterIter": n, "tolUpdate": 0.0,
                  "isPrecond": x0 is not None, "chi_cosmos": _pad_corner(self.chi * self.scale, self.pad), "isXSIM": False, "scale": 1.0}
        if x0 is not None:
            params["precond"] = x0
        with contextlib.redirect_stdout(io.StringIO()):
            out = self.endi.ndiCG(params)
        return out["x"], [float(v) for v in out["rmse_history"]]

    def forward(self, x_pad):
        """Masked field (ppm) of the padded estimate x (radians) with the eNDI kernel."""
        x = torch.from_numpy(np.ascontiguousarray(x_pad, dtype=np.float32)).to(self.device)
        f = torch.fft.ifftn(self.kernel_t * torch.fft.fftn(x)).real.cpu().numpy()
        return f[self.crop] / self.scale * self.mask


def run_variant(variant, phi, solver, stage_a, model_b, mask, weight, ref, voxel, outside, device, rounds, n0, n):
    def clean(f, conv):
        if "A" in variant:
            f = stage_a(f, mask, voxel)[0]
        if "B" in variant:
            f = stage_b(model_b, f, mask, weight, ref, voxel, conv, device)
        return f * mask
    x, hist = solver.solve(clean(phi, outside), None, n0)
    history = [hist[-1]]
    for _ in range(rounds):
        kx = solver.forward(x)
        r = (phi - kx) * mask
        x, hist = solver.solve(kx + clean(r, "out0"), x, n)
        history.append(hist[-1])
    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("runs/generalization_regime"))
    parser.add_argument("--rc2", default="dataRC2")
    parser.add_argument("--rc1-exps", nargs="*", default=["exp1", "exp3", "exp5", "exp6", "exp7", "exp8", "exp9"])
    parser.add_argument("--sim2-exps", nargs="*", default=["sim2_snr100", "sim2_snrcal", "sim2_snr40", "sim2_jumps", "sim2_spheres"])
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--n", type=int, default=4, help="ndiCG maxOuterIter per refinement round (n + 1 steps)")
    parser.add_argument("--ref", type=Path, default=Path("runs/cascade_passive/fixed_meta.json"))
    parser.add_argument("--out", type=Path, default=Path("runs/pnp"))
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    n0 = 25 - args.rounds * (args.n + 1)               # round 0 uses n0 + 1 steps: total 26
    if n0 < 0:
        raise SystemExit("rounds * (n + 1) must not exceed 25")
    if args.dry:
        args.rc1_exps, args.sim2_exps, args.rounds, args.n, n0 = ["exp7"], [], 1, 1, 1
    limit_vram()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    meta_path = args.out / ("fixed_meta_dry.json" if args.dry else "fixed_meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    crop = (lambda a: a[tuple(slice((s - 64) // 2, (s - 64) // 2 + 64) for s in a.shape[:3])]) if args.dry else (lambda a: a)
    model_b = load_noise_clean(device, args.dry)

    def score(tag, exp, phi, chi, mask, weight, ref, voxel, outside):
        solver = Solver(tag, chi, mask, weight, device)
        stage_a = StageA(voxel, device, None, G1_PASSIVE)
        key = f"{tag}/{exp}/AB1"
        if key not in meta:                             # reproduction of the one-pass cascade, 26 steps
            b0 = stage_b(model_b, stage_a(phi, mask, voxel)[0], mask, weight, ref, voxel, outside, device) * mask
            meta[key] = {"nrmse": solver.solve(b0, None, 25)[1][-1:]}
            print(f"{tag} {exp:12s} AB1  {meta[key]['nrmse'][0]:7.2f}%", flush=True)
        for variant in VARIANTS:
            key = f"{tag}/{exp}/PnP-{variant}"
            if key not in meta:
                history = run_variant(variant, phi, solver, stage_a, model_b, mask, weight, ref, voxel, outside, device, args.rounds, n0, args.n)
                meta[key] = {"nrmse": history[-1:], "rounds": history}
                print(f"{tag} {exp:12s} PnP-{variant:2s} {history[-1]:7.2f}%   per round {[round(h, 1) for h in history]}", flush=True)
            meta_path.write_text(json.dumps(meta))

    for tag, exps in (("rc1", args.rc1_exps), ("sim2", args.sim2_exps)):
        if not exps:
            continue
        c = np.load(args.dir / f"{tag}_common.npz")
        chi, mask, ref, wt = (crop(c[k]).astype(np.float32) for k in ("chi_gt", "mask", "clean", "weight"))
        voxel = tuple(float(v) for v in c["voxel_size"])
        for exp in exps:
            z = np.load(args.dir / f"{tag}_{exp}.npz")
            weight = crop(z["weight"]).astype(np.float32) if "weight" in z.files else wt
            score(tag, exp, crop(z["field_dirty"]).astype(np.float32), chi, mask, weight, ref, voxel, "clean")
    d2 = load_rc2(args.rc2, 7.0)
    m2 = d2["mask"].astype(np.float32)
    score("sim2", "sim2_real", crop(d2["field"] * m2), crop(d2["chi"]), crop(m2), crop(d2["weight"]), crop(forward_field(d2["chi"], m2, d2["voxel_size"])),
          d2["voxel_size"], "out0")

    refs = json.loads(args.ref.read_text()) if args.ref.exists() else {}
    at = lambda h: h["nrmse"][min(25, len(h["nrmse"]) - 1)]
    rows = ["eNDI NRMSE %, 26 steps (= 25 fixed iterations). AB1 = one-pass cascade (recomputed); PnP-X = plug-and-play with C = X; refs from cascade_passive: dirty, B, AB",
            f"{'experiment':16s}{'AB1':>9s}{'PnP-AB':>9s}{'PnP-B':>9s}{'PnP-A':>9s}{'ref dirty':>11s}{'ref B':>9s}{'ref AB':>9s}"]
    for key in sorted({k.rsplit("/", 1)[0] for k in meta}):
        g = lambda m: f"{meta[f'{key}/{m}']['nrmse'][-1]:9.2f}" if f"{key}/{m}" in meta else f"{'-':>9s}"
        r = lambda m: f"{at(refs[f'{key}/{m}']):9.2f}" if f"{key}/{m}" in refs else f"{'-':>9s}"
        rows.append(f"{key[key.index('/') + 1:]:16s}{g('AB1')}{g('PnP-AB')}{g('PnP-B')}{g('PnP-A')}{r('dirty'):>11s}{r('B')}{r('AB')}")
    text = "\n".join(rows)
    print(text)
    (args.out / ("table_dry.txt" if args.dry else "table.txt")).write_text(text + "\n")


if __name__ == "__main__":
    main()
