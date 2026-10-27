"""Gate suite 2 (CLAUDE.md 9aq), server, no training: ZERO-TRAINING linear Dipole-let filters scored by eNDI (25 fixed iterations).
The filter is one real gain per segment (least squares, dirty -> clean field inside the mask) of a Dipole-let transform with
 A  3 scales x 3 bands (10 segments)         B  A + coarse split in 3 |D| bands (12)
 C  4 scales x 3 + coarse split (15)         D  5 scales x 3 (16)
Gains for RC1 exp7 / exp8 / exp9 (real tissue / STI / off-diagonal residual) come from the OTHER two real types (leave-one-out, so the
field being scored is never fitted); 'noisefit' gains come from the synthetic noise psi (SNR 60) of RC1 and are applied to exp7-9 and to
the REAL Sim2 field (nothing of Sim2 is fitted). Reference numbers (25 iterations): dirty 93.6 / 67.7 / 83.3, noise_clean net 71.4 / 60.9 / 75.9,
cone notch 0.15 on dirty 64.8 / 52.8 / 60.7 (gate_suite), Sim2 real: dirty 53.94, noise_clean 47.31, oracle 43.10.
Run: MCNO_VRAM_GB=10 nohup .venv/bin/python scripts/gate_suite2.py > runs/gates2.log 2>&1 &     Dry run (laptop): python -I scripts/gate_suite2.py --dry
Writes runs/gates2/results.json (resumable) and table.txt."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_segments import invert  # noqa: E402

from mcno_qsm.artifacts import ArtifactConfig, Volume, make_dirty  # noqa: E402
from mcno_qsm.dipolelets import SIM2_DIPOLELET, DipoleletTransform, create_dipolelet_windows, image_components  # noqa: E402
from mcno_qsm.physics import dipole_kernel  # noqa: E402
from mcno_qsm.rc1 import clean_field, load_rc1, sti_fields  # noqa: E402
from mcno_qsm.rc2 import load_rc2  # noqa: E402
from mcno_qsm.scoring import forward_field, limit_vram  # noqa: E402

CONFIGS = {"A 3x3 (10)": (3, False), "B 3x3+split (12)": (3, True), "C 4x3+split (15)": (4, True), "D 5x3 (16)": (5, False)}
REAL = ("exp7", "exp8", "exp9")


def centre(a, n):
    s = [(m - min(m, n)) // 2 for m in a.shape[:3]]
    return a[tuple(slice(i, i + min(m, n)) for i, m in zip(s, a.shape[:3]))]


def components(field, mask, vox, J, split):
    tr = DipoleletTransform(field.shape, vox, **{**SIM2_DIPOLELET, "scales": J})
    xt = torch.from_numpy(np.ascontiguousarray(field, dtype=np.float32))
    parts = [p.numpy() * mask for p in image_components(tr, xt)]
    if not split:
        return parts
    kern = dipole_kernel(field.shape, vox, dtype=torch.float32)
    thr = np.logspace(np.log10(SIM2_DIPOLELET["delta"]), np.log10(1 / 3), 2).tolist()
    win = create_dipolelet_windows(kern, thr, None, window="gaussian", width_factor=SIM2_DIPOLELET["gaussian_width_factor"])
    spec = list(tr.analysis(xt[None, None]))[-1][0, 0]
    sp = [torch.fft.ifftn(spec * w, dim=(-3, -2, -1), norm="ortho").real.numpy() * mask for w in win]
    return parts[:-1] + sp


def fit(cs, target, sel):
    A = np.stack([c[sel] for c in cs], 1).astype(np.float64)
    return np.linalg.lstsq(A, target[sel].astype(np.float64), rcond=None)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rc1", default="dataRC1")
    parser.add_argument("--rc2", default="dataRC2")
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--dry", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("runs/gates2"))
    args = parser.parse_args()
    if args.dry:
        args.iters = 2
    limit_vram()
    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = args.out / ("results_dry.json" if args.dry else "results.json")
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    crop = (lambda a: centre(a, 64)) if args.dry else (lambda a: a)

    d = load_rc1(args.rc1)
    mask, chi, vox, w = crop(d["mask"]).astype(np.float32), crop(d["chi"]), d["voxel_size"], crop(d["magnitude"])
    b = clean_field(chi, vox, mask)
    psti, pq = sti_fields(crop(d["chi_13"]), crop(d["chi_23"]), crop(d["chi_33"]), vox)
    noisy, _ = make_dirty("noise", Volume.from_rc1({**d, **{k: crop(d[k]) for k in ("chi", "mask", "magnitude", "phs_tissue")}}, "cpu"),
                          np.random.default_rng(0), torch.Generator().manual_seed(0), ArtifactConfig(snr_low=60.0, snr_high=60.0))
    fields = {"noise": noisy.numpy() * mask, "exp7": mask * crop(d["phs_tissue"]), "exp8": mask * psti, "exp9": mask * (b - pq)}
    rc1_set = dict(te=d["te"], b0=d["b0"], pad=64 if args.dry else 192)
    r2 = load_rc2(args.rc2, 7.0)
    m2, c2, w2, v2 = crop(r2["mask"]).astype(np.float32), crop(r2["chi"]), crop(r2["weight"]), r2["voxel_size"]
    b2 = forward_field(c2, m2, v2)
    f2 = crop(r2["field"]) * m2

    def score(key, field, mask_, w_, chi_, settings):
        if key not in cache:
            cache[key] = invert(np.ascontiguousarray(field * mask_, dtype=np.float64), mask_, w_, chi_, args.iters, **settings)
            cache_path.write_text(json.dumps(cache))
            print(f"{key:44s} {cache[key]:7.2f}", flush=True)
        return cache[key]

    for cname, (J, split) in CONFIGS.items():
        comp = {k: components(f, mask, vox, J, split) for k, f in fields.items()}
        gains = {k: fit(comp[k], b, mask > 0) for k in fields}
        for k in REAL:
            others = [o for o in REAL if o != k]
            g = np.mean([gains[o] for o in others], axis=0)
            score(f"{cname}/{k}/loo", sum(gi * c for gi, c in zip(g, comp[k])), mask, w, chi, rc1_set)
            score(f"{cname}/{k}/noisefit", sum(gi * c for gi, c in zip(gains["noise"], comp[k])), mask, w, chi, rc1_set)
        c2c = components(f2, m2, v2, J, split)
        score(f"{cname}/sim2/noisefit", sum(gi * c for gi, c in zip(gains["noise"], c2c)), m2, w2, c2, dict(te=8e-3, b0=7.0, pad=64 if args.dry else 256))
        gr = np.mean([gains[o] for o in REAL], axis=0)
        score(f"{cname}/sim2/realfit", sum(gi * c for gi, c in zip(gr, c2c)), m2, w2, c2, dict(te=8e-3, b0=7.0, pad=64 if args.dry else 256))

    lines = [f"eNDI NRMSE % at {args.iters} fixed iterations of zero-training linear Dipole-let filters",
             "reference (25 it): exp7/8/9 dirty 93.6/67.7/83.3, noise_clean net 71.4/60.9/75.9, notch0.15 on dirty 64.8/52.8/60.7; Sim2 real dirty 53.94, noise_clean 47.31, oracle 43.10",
             f"{'config':20s} " + " ".join(f"{c:>13s}" for c in ("exp7 loo", "exp8 loo", "exp9 loo", "exp7 noisefit", "exp8 noisefit", "exp9 noisefit", "sim2 noisefit", "sim2 realfit"))]
    for cname in CONFIGS:
        keys = [f"{cname}/{k}/loo" for k in REAL] + [f"{cname}/{k}/noisefit" for k in REAL] + [f"{cname}/sim2/noisefit", f"{cname}/sim2/realfit"]
        lines.append(f"{cname:20s} " + " ".join(f"{cache[k]:13.2f}" for k in keys))
    text = "\n".join(lines)
    (args.out / ("table_dry.txt" if args.dry else "table.txt")).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
