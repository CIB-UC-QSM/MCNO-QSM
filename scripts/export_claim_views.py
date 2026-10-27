"""Evidence export for the two claims of CLAUDE.md 9bg and the TV headroom test (no training).

CLAIM 1: a 12-number passive Dipole-let filter (gains in [0, 1], fitted on the MODELLED STI / off-diagonal residuals of RC1 with noise, never on the
measured tissue residual) lowers the eNDI error of real in-vivo-like residuals more than the trained noise network does.
HEADROOM TEST: does a total-variation prior on chi (user's fansiTV, magnitude-weighted data term) on the same fields beat the field-domain floor?

For the RC1 real-residual sets exp7 (measured tissue), exp8 (STI), exp9 (off-diagonal) and the REAL Sim2, for each input
  dirty | B (noise_clean) | A12 (the 12-number passive filter applied everywhere, ungated) | AG (A with the regime gate) | AB (AG then B) | oracle (mask * D chi)
it stores (a) the eNDI chi map at 25 fixed iterations (ndiCG, alpha 1e-5, same inversion as every table of this project) of A12, AG, AB, with their
fields, and (b) a fansiTV sweep over alpha in --alphas (default 5e-5 ... 5e-4), weight = the eNDI magnitude weight, mu1 = 100 alpha, mu2 = 1, 150 ADMM
iterations, padded exactly as eNDI (192^3 RC1, 256^3 Sim2); the NRMSE / XSIM history of every run, and the chi map of the best alpha per (set, input).
The best alpha is chosen with the ground truth (ORACLE-TUNED, optimistic) and is chosen separately for RC1 and Sim2 because their data scales differ; the
dirty-input alpha is also reported so that a fixed-alpha reading is possible. Sim2 is tuned on Sim2's own ground truth: a diagnostic, not a design input.

Run (server): MCNO_VRAM_GB=10 nohup .venv/bin/python scripts/export_claim_views.py > runs/claim_views.log 2>&1 &
Dry run (laptop, 64^3 crop, untrained B, 3 ADMM iterations): python -I scripts/export_claim_views.py --dry
Writes runs/claim_views/meta.json (resumable) and runs/claim_views/maps.npz (float16, cropped to the brain box + 4 voxels; keys <set>__<name>).
"""

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
from cascade_gate import DIP, G1_PASSIVE, SETTINGS, StageA, load_noise_clean, stage_b  # noqa: E402
from export_regime_views import bounding_box  # noqa: E402

from mcno_qsm.rc2 import load_rc2, phase_scale  # noqa: E402
from mcno_qsm.scoring import _modules, _pad_corner, forward_field, limit_vram  # noqa: E402


def padded_params(field, chi, mask, weight, tag, extra):
    te, b0, pad = SETTINGS[tag]
    tools, _ = _modules()
    scale = phase_scale(te, b0)
    params = {"K": tools.dipole_kernel((pad,) * 3, (pad,) * 3), "input": _pad_corner(field * scale, pad), "isNonlinear": False,
              "weight": _pad_corner(weight, pad), "mask": _pad_corner(mask, pad), "tolUpdate": 0.0,
              "chi_cosmos": _pad_corner(chi * scale, pad), "isXSIM": True, "scale": 1.0, **extra}
    return params, scale, tuple(slice(0, n) for n in field.shape)


def endi_chi(field, chi, mask, weight, tag, iters=25):
    """-> (chi map in ppm, NRMSE history %). Same call as rescore_fixed.invert, but the map is returned."""
    params, scale, crop = padded_params(field, chi, mask, weight, tag, {"alpha": 1e-5, "maxOuterIter": iters, "isPrecond": False})
    _, endi = _modules()
    with contextlib.redirect_stdout(io.StringIO()):
        out = endi.ndiCG(params)
    return out["x"][crop] / scale, [float(v) for v in out["rmse_history"]]


def tv_chi(field, chi, mask, weight, tag, alpha, iters):
    """fansiTV (linear, magnitude-weighted data term, mu1 = 100 alpha, mu2 = 1) -> (chi map in ppm, NRMSE history %, XSIM history)."""
    params, scale, crop = padded_params(field, chi, mask, weight, tag, {"alpha": alpha, "maxOuterIter": iters})
    _modules()
    from FANSI_module_torch import fansiTV
    with contextlib.redirect_stdout(io.StringIO()):
        out = fansiTV(params)
    return out["x"][crop] / scale, [float(v) for v in out["rmse_history"]], [float(v) for v in out.get("xsim_history", [])]


@torch.no_grad()
def a12(stage, field, mask, voxel):
    """The 12-number passive filter with NO gate: mask * (x - sum_s (1 - g_s) comp_s)."""
    x = torch.from_numpy(np.ascontiguousarray(field * mask, np.float32)).to(stage.device)[None, None]
    transform, _, _ = stage.model._geometry(x.shape[-3:], x.device, voxel)
    comps = torch.stack([torch.fft.ifftn(s, dim=(-3, -2, -1), norm="ortho").real[:, 0] for s in transform.analysis(x)], 1)
    full = torch.einsum("s,bsxyz->bxyz", stage.removal1, comps)[:, None]
    m = torch.from_numpy(mask).to(stage.device)[None, None]
    return (m * (x - full))[0, 0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path("runs/generalization_regime"))
    parser.add_argument("--rc2", default="dataRC2")
    parser.add_argument("--rc1-exps", nargs="*", default=["exp7", "exp8", "exp9"])
    parser.add_argument("--alphas", nargs="*", type=float, default=[5e-5, 1e-4, 2e-4, 3e-4, 5e-4])
    parser.add_argument("--tv-iters", type=int, default=150)
    parser.add_argument("--out", type=Path, default=Path("runs/claim_views"))
    parser.add_argument("--no-b", action="store_true", help="skip the noise_clean network (no B / AB; needs no checkpoint): the laptop run of the 12-number filter")
    parser.add_argument("--no-tv", action="store_true", help="skip the fansiTV sweep")
    parser.add_argument("--dry", action="store_true", help="laptop smoke test: 64^3 crop, untrained B, 3 ADMM iterations, 2 iterations of eNDI, exp7 only")
    args = parser.parse_args()
    iters_endi = 25
    if args.dry:
        args.dir, args.rc1_exps, args.tv_iters, iters_endi, args.alphas = Path("runs/generalization"), ["exp7"], 3, 2, args.alphas[:2]
    limit_vram()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)
    meta_path = args.out / ("meta_dry.json" if args.dry else "meta.json")
    maps_path = args.out / ("maps_dry.npz" if args.dry else "maps.npz")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    maps = dict(np.load(maps_path)) if maps_path.exists() else {}
    crop64 = (lambda a: a[tuple(slice((n - 64) // 2, (n - 64) // 2 + 64) for n in a.shape[:3])]) if args.dry else (lambda a: a)
    model_b = None if args.no_b else load_noise_clean(device, args.dry)

    def run_set(tag, exp, dirty, chi, mask, weight, reference, voxel, outside):
        box = bounding_box(mask)
        put = lambda name, vol: maps.__setitem__(f"{exp}__{name}", np.ascontiguousarray(vol[box]).astype(np.float16))
        stage = StageA(voxel, device, None, G1_PASSIVE)
        fields = {"dirty": dirty * mask, "oracle": reference * mask}
        fields["A12"] = a12(stage, dirty, mask, voxel)
        fields["AG"] = stage(dirty, mask, voxel)[0]
        if model_b is not None:
            fields["B"] = stage_b(model_b, dirty, mask, weight, reference, voxel, outside, device)
            fields["AB"] = stage_b(model_b, fields["AG"], mask, weight, reference, voxel, outside, device)
        put("chi_gt", chi * mask)
        put("mask", mask)
        for name, f in fields.items():
            put(f"field_{name}", f)
        for name in [n for n in ("A12", "AG", "AB", "dirty", "B", "oracle") if n in fields]:          # eNDI at 25 fixed iterations
            key = f"endi/{exp}/{name}"
            if key not in meta or f"{exp}__chi_endi_{name}" not in maps:
                x, hist = endi_chi(fields[name].astype(np.float32), chi, mask, weight, tag, iters_endi)
                meta[key] = {"nrmse": hist}
                put(f"chi_endi_{name}", x * mask)
                meta_path.write_text(json.dumps(meta))
                np.savez_compressed(maps_path, **maps)
                print(f"{exp:10s} eNDI {name:7s} @{len(hist)}: {hist[-1]:6.2f}%", flush=True)
        for name in [n for n in ("dirty", "B", "A12", "AB", "oracle") if n in fields and not args.no_tv]:               # TV sweep
            best = None
            for alpha in args.alphas:
                key = f"tv/{exp}/{name}/{alpha:g}"
                if key not in meta:
                    x, hist, xs = tv_chi(fields[name].astype(np.float32), chi, mask, weight, tag, alpha, args.tv_iters)
                    meta[key] = {"nrmse": hist, "xsim": xs}
                    meta_path.write_text(json.dumps(meta))
                    maps[f"{exp}__tmp_{name}_{alpha:g}"] = (x * mask).astype(np.float16)    # kept until the best alpha is known
                    print(f"{exp:10s} TV {name:7s} alpha {alpha:7.0e}: {hist[-1]:6.2f}% (min {min(hist):6.2f}% @{int(np.argmin(hist))})", flush=True)
                final = meta[key]["nrmse"][-1]
                if best is None or final < best[0]:
                    best = (final, alpha)
            meta[f"tv_best/{exp}/{name}"] = {"alpha": best[1], "nrmse": best[0]}
            tmp = maps.get(f"{exp}__tmp_{name}_{best[1]:g}")
            if tmp is None:                    # resumed after the temporary map was lost: recompute the best alpha only
                tmp = tv_chi(fields[name].astype(np.float32), chi, mask, weight, tag, best[1], args.tv_iters)[0] * mask
            put(f"chi_tv_{name}", tmp.astype(np.float32))
            for alpha in args.alphas:
                maps.pop(f"{exp}__tmp_{name}_{alpha:g}", None)
            meta_path.write_text(json.dumps(meta))
            np.savez_compressed(maps_path, **maps)

    # ---- RC1 real-residual sets
    if (args.dir / "rc1_common.npz").exists():
        c = np.load(args.dir / "rc1_common.npz")
        chi, mask, ref, wt = (crop64(c[k]).astype(np.float32) for k in ("chi_gt", "mask", "clean", "weight"))
        voxel = tuple(float(v) for v in c["voxel_size"])
        for exp in args.rc1_exps:
            z = np.load(args.dir / f"rc1_{exp}.npz")
            weight = crop64(z["weight"]).astype(np.float32) if "weight" in z.files else wt
            run_set("rc1", exp, crop64(z["field_dirty"]).astype(np.float32), chi, mask, weight, ref, voxel, "clean")
    # ---- real Sim2
    if not args.dry:
        d2 = load_rc2(args.rc2, 7.0)
        m2 = d2["mask"].astype(np.float32)
        run_set("sim2", "sim2_real", d2["field"] * m2, d2["chi"], m2, d2["weight"], forward_field(d2["chi"], m2, d2["voxel_size"]), d2["voxel_size"], "out0")
    np.savez_compressed(maps_path, **maps)
    print("wrote", maps_path, meta_path, flush=True)


if __name__ == "__main__":
    main()
