"""Realism check of the intravoxel-dephasing simulator against Sim2's calcification (CPU, no training).

Sim2's source (voxels with Chi < -0.5, plus its partial-volume edge) is replaced by a uniform ellipsoid of the SAME total moment and the same
second moments, simulated on a fine grid (`--factors`), averaged back to the coarse grid for the four echoes of Magnitude.nii, noised, and the
multi-echo frequency is estimated. The simulated radial profiles (true-field rms, psi rms, kept fraction of the true field, magnitude ratio per
echo) are printed next to the measured ones. Variants: fine factor, intrinsic signal void inside the source, frequency estimator.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mcno_qsm.intravoxel import dephase, source_chi_fine  # noqa: E402
from mcno_qsm.rc2 import ECHO_TIMES, load_rc2  # noqa: E402
from mcno_qsm.scoring import forward_field  # noqa: E402

B0 = 7.0
GAMMA = 42.58
EDGES = [0, 1, 1.5, 2, 3, 4, 6, 8, 12, 1e9]
HALF = 24


def source_model(chi, voxel):
    """Source voxels (Chi < -0.5 and its negative partial-volume edge), moment, centroid and radii of the equivalent uniform ellipsoid."""
    core = chi < -0.5
    near = ndi.binary_dilation(core, iterations=3)
    region = near & (chi < -0.05)
    weights = -chi * region
    total = float(chi[region].sum())
    idx = np.argwhere(region).astype(float)
    w = weights[region]
    centre = (idx * w[:, None]).sum(0) / w.sum()
    cov = ((idx - centre).T * w) @ (idx - centre) / w.sum()
    eigval, eigvec = np.linalg.eigh(cov)
    radii = np.sqrt(5.0 * eigval)
    return region, total, centre, radii, eigvec


def estimate_frequency(signal, tes, mode):
    """Hz from a (…, E) complex array: weighted least-squares slope of the phase unwrapped along the echoes."""
    phase = np.unwrap(np.angle(signal), axis=-1)
    w = np.abs(signal) ** 2 + 1e-12
    t = np.asarray(tes)
    if mode == "first":
        return phase[..., 0] / (2 * np.pi * t[0])
    if mode == "slope0":   # line through the origin
        return (w * phase * t).sum(-1) / ((w * t * t).sum(-1)) / (2 * np.pi)
    tm = (w * t).sum(-1) / w.sum(-1)
    pm = (w * phase).sum(-1) / w.sum(-1)
    num = (w * (t - tm[..., None]) * (phase - pm[..., None])).sum(-1)
    den = (w * (t - tm[..., None]) ** 2).sum(-1)
    return num / den / (2 * np.pi)


def run(args):
    d = load_rc2(ROOT / "dataRC2")
    chi, field, mag, mask, vs = d["chi"], d["field"], d["magnitude"], d["mask"], d["voxel_size"]
    voxel = vs[0]
    region, total, centre, radii, eigvec = source_model(chi, voxel)
    radii = radii * args.shrink
    volume = 4 / 3 * np.pi * np.prod(radii)
    amplitude = total / volume
    print(f"source: {region.sum()} voxels, moment {total:.2f} ppm*voxel, centre {np.round(centre, 1)}, radii {np.round(radii, 2)} voxels,"
          f" uniform-ellipsoid amplitude {amplitude:.2f} ppm; principal axes (rows): {np.round(eigvec.T, 2).tolist()}")
    chi_bg = chi.copy()
    chi_bg[region] = 0.0
    b_bg = forward_field(chi_bg, None, vs)
    b_true = forward_field(chi, None, vs)
    lo = np.round(centre).astype(int) - HALF
    sl = tuple(slice(l, l + 2 * HALF) for l in lo)
    local_centre = centre - lo
    # Sim2 reference (measured)
    m = mag / mag.max()
    src_core = chi < -0.5
    dist_all = ndi.distance_transform_edt(~src_core, sampling=vs)
    base = np.array([np.median(m[..., e][mask & (dist_all > 15)]) for e in range(4)])
    tes = np.array(ECHO_TIMES)
    per_ppm_s = 2 * np.pi * GAMMA * B0   # rad / (ppm s)
    bg_t = torch.from_numpy(b_bg[sl].astype(np.float64))
    rng = np.random.default_rng(args.seed)
    ref = {"b": b_true[sl], "dirty": field[sl], "mask": mask[sl], "mag": m[sl] / base, "dist": dist_all[sl]}
    results = {}
    for factor in args.factors:
        # principal axes: use the eigenvector rotation only when the axes are not aligned with the grid (they are here to within a few degrees)
        for void in (False, True):
            chi_f = source_chi_fine((2 * HALF,) * 3, factor, local_centre, radii, amplitude)
            if void:   # no protons inside the source (optionally a void larger than the susceptibility ellipsoid)
                dip = (source_chi_fine((2 * HALF,) * 3, factor, local_centre, radii * args.void_grow, 1.0) == 0).double()
            else:
                dip = None
            signals, target = [], None
            for e, te in enumerate(tes):
                mag_e = torch.full_like(bg_t, float(base[e]))
                s, tgt = dephase(bg_t, mag_e, chi_f, factor, voxel, float(te), per_ppm_s, intrinsic_dip=dip)
                signals.append(s.numpy())
                target = tgt.numpy()
            clean_signal = np.stack(signals, -1)
            # noise calibrated so that psi rms far from the source matches the measured 0.0032 ppm (bisection on sigma)
            far = ref["mask"] & (ref["dist"] > 10)
            lo_s, hi_s = 1e-5, 1.0
            for _ in range(16):
                sigma = np.sqrt(lo_s * hi_s)
                noisy = clean_signal + sigma * (rng.standard_normal(clean_signal.shape) + 1j * rng.standard_normal(clean_signal.shape))
                est = estimate_frequency(noisy, tes, args.estimator) / (GAMMA * B0)
                rms = np.sqrt(((est - target)[far] ** 2).mean())
                lo_s, hi_s = (sigma, hi_s) if rms < args.far_psi else (lo_s, sigma)
            sigma = np.sqrt(lo_s * hi_s)
            noisy = clean_signal + sigma * (rng.standard_normal(clean_signal.shape) + 1j * rng.standard_normal(clean_signal.shape))
            est = estimate_frequency(noisy, tes, args.estimator) / (GAMMA * B0)
            mag_sim = np.abs(noisy) / base
            results[(factor, void)] = (target, est, mag_sim, sigma)
    return ref, results, total, centre, local_centre


def table(name, b, dirty, mag_ratio, mask, dist):
    print(f"\n{name}\n  dist mm     n   rms_b  rms_psi  keep | magnitude ratio e1 e2 e3 e4")
    rows = []
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        s = mask & (dist >= lo) & (dist < hi)
        if not s.any():
            continue
        keep = float((dirty[s] * b[s]).sum() / (b[s] ** 2).sum())
        rows.append((lo, hi, int(s.sum()), float(np.sqrt((b[s] ** 2).mean())), float(np.sqrt(((dirty - b)[s] ** 2).mean())), keep,
                     [float(np.median(mag_ratio[..., e][s])) for e in range(4)]))
        r = rows[-1]
        print(f"  {r[0]:4.1f}-{min(r[1], 99):4.1f} {r[2]:5d}  {r[3]:.4f}  {r[4]:.4f} {r[5]:5.2f} | " + " ".join(f"{x:4.2f}" for x in r[6]))
    return rows


def summarize(args, ref, results):
    bins = [(0, 1), (1, 1.5), (1.5, 2), (2, 3), (3, 4), (4, 6)]

    def line(name, b, dirty, mag):
        out = []
        for lo, hi in bins:
            s = ref["mask"] & (ref["dist"] >= lo) & (ref["dist"] < hi)
            keep = (dirty[s] * b[s]).sum() / (b[s] ** 2).sum()
            out.append(f"{np.sqrt(((dirty - b)[s] ** 2).mean()):.4f}/{keep:5.2f}/{np.median(mag[..., 0][s]):.2f}/{np.median(mag[..., 3][s]):.2f}")
        print(f"{name:34s}", "  ".join(out))
    print(f"{'rms_psi/keep/mag_e1/mag_e4 by mm':34s}", "  ".join(f"{lo}-{hi}mm".rjust(21) for lo, hi in bins))
    line("measured Sim2", ref["b"], ref["dirty"], ref["mag"])
    for (factor, void), (target, est, mag_sim, sigma) in results.items():
        line(f"f{factor} void={int(void)} shrink={args.shrink} grow={args.void_grow} {args.estimator}", target, est, mag_sim)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--factors", type=int, nargs="+", default=[3, 4])
    p.add_argument("--estimator", choices=["slope", "slope0", "first"], default="slope")
    p.add_argument("--far-psi", type=float, default=0.0032, help="measured psi rms (ppm) 8-25 mm from the source")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shrink", type=float, default=1.0, help="scale the ellipsoid radii (amplitude rises as shrink^-3, moment fixed)")
    p.add_argument("--void-grow", type=float, default=1.0, help="proton-free void radii relative to the susceptibility ellipsoid")
    p.add_argument("--summary", action="store_true", help="print only the compact comparison")
    args = p.parse_args()
    ref, results, total, centre, local_centre = run(args)
    if args.summary:
        summarize(args, ref, results)
        return
    table("MEASURED Sim2 (dirty = Frequency.nii, b = D*Chi)", ref["b"], ref["dirty"], ref["mag"], ref["mask"], ref["dist"])
    for (factor, void), (target, est, mag_sim, sigma) in results.items():
        # distance from the model source: coarse voxels whose block-mean chi is below -0.5 are not stored; use the measured source distance
        # (same centre, same moment), so the bins refer to the same voxels as in the measured table
        table(f"SIMULATED factor {factor}, intrinsic void {void}, estimator {args.estimator}, noise sigma {sigma:.4f} (target = block-mean field)",
              target, est, mag_sim, ref["mask"], ref["dist"])
    # axis profiles through the source centre
    c = tuple(int(round(x)) for x in local_centre)
    np.set_printoptions(linewidth=220, precision=3, suppress=True)
    for ax in range(3):
        sel = lambda a: np.array([a[tuple(c[k] + (i if k == ax else 0) for k in range(3))] for i in range(-6, 7)])
        print(f"\nprofile along axis {ax} (offsets -6..6 voxels)")
        print("  measured true b  ", sel(ref["b"]))
        print("  measured dirty   ", sel(ref["dirty"]))
        for (factor, void), (target, est, mag_sim, sigma) in results.items():
            print(f"  sim f{factor} void={int(void)} tgt ", sel(target))
            print(f"  sim f{factor} void={int(void)} dirty", sel(est))


if __name__ == "__main__":
    main()
