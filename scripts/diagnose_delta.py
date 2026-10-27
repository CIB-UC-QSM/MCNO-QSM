"""Is delta = 0.05 a good Dipole-let threshold? Label-light diagnostic on the linear part of the block (CPU, no training).

delta only sets the edge of the |D| < delta "core" band: with base_angular_bands = 3 the |D| thresholds of every scale are
[delta, 1/3]. For each delta we build the 10-segment transform, split the dirty and the clean Sim2 field into segments and fit
ONE REAL GAIN PER SEGMENT by least squares (the best a block could do with scalar gains and no nonlinearity or spatial branch).
Two fits are reported, both scored on Sim2 (masked phase NRMSE after applying the gains to the dirty field):
  oracle   gains fitted on Sim2 itself (optimistic, target-tuned upper bound)
  transfer gains fitted on RC1 (clean = mask * D*chi_cosmos, dirty = clean + white noise with Sim2's psi rms), applied to Sim2
Also printed per delta: k-space share of the core, and the share of clean / psi power inside it.
Run (project root): .venv/bin/python scripts/diagnose_delta.py --data dataRC2 --rc1 dataRC1
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from mcno_qsm.dipolelets import SIM2_DIPOLELET, DipoleletTransform
from mcno_qsm.physics import dipole_kernel
from mcno_qsm.rc1 import load_rc1
from mcno_qsm.rc2 import load_rc2
from mcno_qsm.scoring import forward_field


def segment_spectra(transform, volume):
    x = torch.from_numpy(np.ascontiguousarray(volume, dtype=np.float32))[None, None]
    with torch.no_grad():
        return [s[0, 0] for s in transform.analysis(x)]


def apply_gains(segments, gains, mask):
    total = sum(g * s for g, s in zip(gains, segments))
    return torch.fft.ifftn(total, norm="ortho").real.numpy() * mask


def nrmse(estimate, reference, mask):
    return float(100 * np.linalg.norm(mask * (estimate - reference)) / np.linalg.norm(mask * reference))


def solve_gains(dirty_segments, clean_spectrum):
    n = len(dirty_segments)
    flat = [d.flatten() for d in dirty_segments]
    target = clean_spectrum.flatten()
    gram = np.array([[torch.real(torch.vdot(flat[i], flat[j])).item() for j in range(n)] for i in range(n)])
    rhs = np.array([torch.real(torch.vdot(flat[i], target)).item() for i in range(n)])
    return np.linalg.solve(gram + 1e-12 * np.trace(gram) / n * np.eye(n), rhs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataRC2")
    parser.add_argument("--rc1", default="dataRC1")
    parser.add_argument("--deltas", type=float, nargs="+", default=[0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3])
    parser.add_argument("--out", type=Path, default=Path("runs/diagnose_delta.json"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)

    sim = load_rc2(args.data)
    mask, voxel = sim["mask"], sim["voxel_size"]
    clean = forward_field(sim["chi"], mask, voxel)
    dirty = sim["field"] * mask
    psi = dirty - clean
    rms_psi = float(np.sqrt((psi[mask] ** 2).mean()))

    rc1 = load_rc1(args.rc1)
    mask1, voxel1 = rc1["mask"], rc1["voxel_size"]
    clean1 = forward_field(rc1["chi"], mask1, voxel1)
    rng = np.random.default_rng(args.seed)
    dirty1 = clean1 + (rng.standard_normal(clean1.shape) * rms_psi * mask1).astype(np.float32)

    print(f"Sim2 dirty phase NRMSE {nrmse(dirty, clean, mask):.2f}%  | psi rms {rms_psi:.4f} ppm | RC1 uses white noise of that rms")
    print(f"{'delta':>6s} {'core k-share':>12s} {'clean in core':>13s} {'psi in core':>11s} | {'oracle fit':>10s} {'RC1 transfer':>12s}")
    results = {}
    clean_spec = torch.fft.fftn(torch.from_numpy(clean), norm="ortho")
    for delta in args.deltas:
        params = {**SIM2_DIPOLELET, "delta": delta}
        tf_sim = DipoleletTransform(dirty.shape, voxel, **params)
        tf_rc1 = DipoleletTransform(dirty1.shape, voxel1, **params)
        segs_dirty = segment_spectra(tf_sim, dirty)
        gains_oracle = solve_gains(segs_dirty, clean_spec)
        out_oracle = nrmse(apply_gains(segs_dirty, gains_oracle, mask), clean, mask)
        segs1 = segment_spectra(tf_rc1, dirty1)
        gains_rc1 = solve_gains(segs1, torch.fft.fftn(torch.from_numpy(clean1), norm="ortho"))
        out_transfer = nrmse(apply_gains(segs_dirty, gains_rc1, mask), clean, mask)
        # core band = |D| < delta (continuous bound; the Gaussian window is soft around it)
        d_abs = dipole_kernel(dirty.shape, voxel, dtype=torch.float32).abs()
        core = d_abs < delta
        power = lambda v: (torch.fft.fftn(torch.from_numpy(v), norm="ortho").abs() ** 2)
        share = lambda v: float(power(v)[core].sum() / power(v).sum())
        results[delta] = {"core_k_share": float(core.float().mean()), "clean_in_core": share(clean), "psi_in_core": share(psi),
                          "oracle_fit_nrmse": out_oracle, "rc1_transfer_nrmse": out_transfer,
                          "gains_oracle": gains_oracle.round(3).tolist(), "gains_rc1": gains_rc1.round(3).tolist()}
        r = results[delta]
        print(f"{delta:6.3f} {100 * r['core_k_share']:11.1f}% {100 * r['clean_in_core']:12.2f}% {100 * r['psi_in_core']:10.2f}% | "
              f"{out_oracle:9.2f}% {out_transfer:11.2f}%", flush=True)

    # references: one global gain (no segmentation) and radial scales only (no |D| bands)
    one = solve_gains([torch.fft.fftn(torch.from_numpy(dirty), norm="ortho")], clean_spec)
    print(f"\nreference, one global gain g = {one[0]:.3f}: NRMSE {nrmse(one[0] * dirty, clean, mask):.2f}%")
    radial = DipoleletTransform(dirty.shape, voxel, **{**SIM2_DIPOLELET, "base_angular_bands": 1})
    segs = segment_spectra(radial, dirty)
    print(f"reference, radial scales only ({len(segs)} segments, no |D| bands): oracle fit NRMSE "
          f"{nrmse(apply_gains(segs, solve_gains(segs, clean_spec), mask), clean, mask):.2f}%")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
