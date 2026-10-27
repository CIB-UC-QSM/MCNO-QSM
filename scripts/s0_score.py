"""S0: eNDI (ndiCG_auto) on the dirty field, the oracle and linear baselines, identical settings for all.

Baselines are tuned against the clean target (best case for a linear filter), not blind.
Run on a GPU (256^3 padding): uv run python -I scripts/s0_score.py --data dataRC2 --out runs/s0/score.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from mcno_qsm.rc2 import load_rc2
from mcno_qsm.scoring import forward_field, masked_nrmse, score_endi


def masked_lowpass(field, mask, sigma):
    """Normalized convolution so the zeros outside the mask do not bleed in."""
    weights = gaussian_filter(mask.astype(np.float32), sigma)
    return gaussian_filter(field * mask, sigma) / np.maximum(weights, 1e-3) * mask


def radial_wiener(field, clean, mask, bins=64):
    """H(|k|) = Sc / (Sc + Sn) from radially averaged oracle spectra; Sn = spectrum of (field - clean)."""
    freqs = [np.fft.fftfreq(n) for n in field.shape]
    radius = np.sqrt(sum(f**2 for f in np.meshgrid(*freqs, indexing="ij")))
    index = np.minimum((radius / radius.max() * bins).astype(int), bins - 1)
    sc = np.abs(np.fft.fftn(clean)) ** 2
    sn = np.abs(np.fft.fftn(field - clean)) ** 2
    count = np.bincount(index.ravel(), minlength=bins)
    prof = lambda s: np.bincount(index.ravel(), s.ravel(), minlength=bins) / np.maximum(count, 1)
    h = prof(sc) / (prof(sc) + prof(sn) + 1e-30)
    return np.real(np.fft.ifftn(np.fft.fftn(field) * h[index])) * mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="dataRC2")
    parser.add_argument("--out", default="runs/s0/score.json")
    parser.add_argument("--te", type=float, default=8e-3)
    parser.add_argument("--b0", type=float, default=7.0)
    args = parser.parse_args()

    data = load_rc2(args.data, args.b0)
    mask, chi, weight = data["mask"], data["chi"], data["weight"]
    clean = forward_field(chi, mask)
    dirty = data["field"] * mask
    sigmas = [0.25, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    best_sigma = min(sigmas, key=lambda s: masked_nrmse(masked_lowpass(dirty, mask, s), clean, mask))
    inputs = {
        "dirty_masked": dirty,
        "dirty_unmasked_notebook": data["field"],
        "oracle": clean,
        f"lowpass_sigma{best_sigma}": masked_lowpass(dirty, mask, best_sigma),
        "radial_wiener_oracle": radial_wiener(dirty, clean, mask),
    }
    report = {}
    for name, field in inputs.items():
        result = score_endi(field, chi, mask, weight, te=args.te, b0=args.b0)
        report[name] = {
            "phase_nrmse_vs_clean_pct": masked_nrmse(field, clean, mask),
            "endi_nrmse_pct": result["nrmse"], "endi_xsim": result["xsim"],
            "stop_iteration": result["iterations"], "seconds": result["seconds"],
            "nrmse_history": result["nrmse_history"], "xsim_history": result["xsim_history"],
        }
        print(f"{name:28s} phase {report[name]['phase_nrmse_vs_clean_pct']:7.2f}%  eNDI {result['nrmse']:7.2f}%  "
              f"XSIM {result['xsim']:.4f}  stop {result['iterations']}")
    report["settings"] = {"te": args.te, "b0": args.b0, "alpha": 1e-5, "max_iter": 150, "pad": 256,
                          "target": "mask * A(Chi), FANSI kernel, unpadded", "chi_gt": "Chi.nii (unmasked)"}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
