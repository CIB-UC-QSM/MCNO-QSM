"""S0: reproduce the CLAUDE.md 3.3 numbers on RC2 (units, psi size and spectrum). No training.

Run: uv run python -I scripts/s0_sim2_diagnostics.py --data dataRC2 --out runs/s0/diagnostics.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt

from mcno_qsm.rc2 import hz_to_ppm, load_rc2
import torch

from mcno_qsm.physics import dipole_kernel
from mcno_qsm.scoring import forward_field


def energy_fraction(x, selector):
    power = np.abs(np.fft.fftn(x)) ** 2
    return float(power[selector].sum() / power.sum())


def roughness(x, mask):
    gx = np.stack(np.gradient(x))
    return float(np.sqrt((gx[:, mask] ** 2).mean()) / x[mask].std())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataRC2")
    parser.add_argument("--out", default="runs/s0/diagnostics.json")
    parser.add_argument("--b0", type=float, default=7.0)
    args = parser.parse_args()

    data = load_rc2(args.data, args.b0)
    mask, chi, field = data["mask"], data["chi"], data["field"]
    shape = chi.shape
    kernel = dipole_kernel(shape, (1.0, 1.0, 1.0), dtype=torch.float64).numpy()
    clean = forward_field(chi, mask)
    dirty = field * mask
    psi = dirty - clean

    slope, intercept = np.polyfit(clean[mask], dirty[mask], 1)
    freqs = [np.fft.fftfreq(n) for n in shape]
    radius = np.sqrt(sum(f**2 for f in np.meshgrid(*freqs, indexing="ij")))
    cone = np.abs(kernel) < 0.03
    distance = distance_transform_edt(mask)
    ratios = {}
    for lo, hi in ((1, 3), (3, 6), (6, 12), (12, 1e9)):
        shell = mask & (distance >= lo) & (distance < hi)
        ratios[f"{lo}-{hi}"] = float(psi[shell].std() / clean[shell].std())
    report = {
        "units": {"slope_dirty_vs_clean": float(slope), "intercept": float(intercept),
                  "corr": float(np.corrcoef(clean[mask], dirty[mask])[0, 1]),
                  "hz_per_ppm_expected": 42.58 * args.b0},
        "psi_over_dirty_norm": float(np.linalg.norm(psi) / np.linalg.norm(dirty)),
        "rms_psi_ppm": float(psi[mask].std()), "rms_clean_ppm": float(clean[mask].std()),
        "psi_energy_fraction": {"k>0.2": energy_fraction(psi, radius > 0.2),
                                "k<0.02": energy_fraction(psi, radius < 0.02),
                                "|D|<0.03": energy_fraction(psi, cone)},
        "cone_energy_fraction": {"dirty": energy_fraction(dirty, cone), "clean": energy_fraction(clean, cone)},
        "cone_kspace_share": float(cone.mean()),
        "psi_over_clean_rms_by_distance_to_edge_vox": ratios,
        "roughness": {"psi": roughness(psi, mask), "clean": roughness(clean, mask)},
        "frequency_outside_mask_max_abs_ppm": float(np.abs(field[~mask]).max()),
        "chi_outside_mask_nonzero_fraction": float((chi[~mask] != 0).mean()),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
