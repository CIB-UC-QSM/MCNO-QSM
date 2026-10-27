"""What does each cleaner remove on Sim2? Per-band comparison of the removed component with the true psi (CPU, no GPU).

For every run: removed = dirty - cleaned (both masked, ppm), psi = dirty - mask*D*chi (the true non-dipolar part).
Per band of |k| (cycles/mm) and of |D|, from the Fourier transforms of the masked volumes:
  capture      Re<psi, removed> / <psi, psi>        1 = removes exactly the psi in that band, 0 = nothing, <0 wrong sign
  coherence    Re<psi, removed> / (|psi| |removed|)  how well the removed shape matches psi, independent of its amplitude
  error ratio  |psi - removed|^2 / |psi|^2          <1 = band got cleaner than dirty, >1 = error added
  signal loss  Re<clean, removed> / <clean, clean>   fraction of the TRUE field removed (should be ~0; large = harms the inversion)
Run: .venv/bin/python scripts/diagnose_removed.py --runs noise mix tissue combo --out runs/diagnosis
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

from mcno_qsm.rc2 import load_rc2  # noqa: E402
from mcno_qsm.scoring import forward_field  # noqa: E402

K_EDGES = [0.0, 0.05, 0.1, 0.2, 0.3, 10.0]
D_EDGES = [0.0, 0.03, 0.15, 0.3, 10.0]


def band_stats(psi_f, removed_f, clean_f, selections):
    rows = []
    for name, sel in selections:
        p, r, c = psi_f[sel], removed_f[sel], clean_f[sel]
        pp, rr, cc = (np.abs(p) ** 2).sum(), (np.abs(r) ** 2).sum(), (np.abs(c) ** 2).sum()
        cross = np.real(np.vdot(p, r))
        rows.append({"band": name, "psi_power_share": float(pp / (np.abs(psi_f) ** 2).sum()),
                     "capture": float(cross / pp), "coherence": float(cross / np.sqrt(pp * rr + 1e-30)),
                     "error_ratio": float((np.abs(p - r) ** 2).sum() / pp),
                     "signal_loss": float(np.real(np.vdot(c, r)) / cc)})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="+", default=["noise", "mix", "tissue", "combo"])
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"))
    parser.add_argument("--data", default="dataRC2")
    parser.add_argument("--out", type=Path, default=Path("runs/diagnosis"))
    args = parser.parse_args()

    data = load_rc2(args.data)
    mask, chi, voxel = data["mask"], data["chi"], data["voxel_size"]
    clean = forward_field(chi, mask, voxel)
    dirty = data["field"] * mask
    freqs = [np.fft.fftfreq(n, d=v) for n, v in zip(mask.shape, voxel)]
    kx, ky, kz = np.meshgrid(*freqs, indexing="ij")
    k2 = kx**2 + ky**2 + kz**2
    radius = np.sqrt(k2)
    dip = np.abs(1 / 3 - kz**2 / np.where(k2 == 0, 1, k2))
    selections = [(f"|k| {lo:g}-{hi:g}" if hi < 10 else f"|k| >{lo:g}", (radius >= lo) & (radius < hi)) for lo, hi in zip(K_EDGES[:-1], K_EDGES[1:])]
    selections += [(f"|D| {lo:g}-{hi:g}" if hi < 10 else f"|D| >{lo:g}", (dip >= lo) & (dip < hi)) for lo, hi in zip(D_EDGES[:-1], D_EDGES[1:])]
    psi_f, clean_f = np.fft.fftn(dirty - clean), np.fft.fftn(clean)

    report = {}
    for name in args.runs:
        cleaned = nib.load(args.runs_dir / f"s4b_{name}" / "best_clean.nii.gz").get_fdata(dtype=np.float32) * mask
        report[name] = band_stats(psi_f, np.fft.fftn(dirty - cleaned), clean_f, selections)
        print(f"\n{name}: band, share of psi power, capture, coherence, error ratio (<1 better), signal loss")
        for row in report[name]:
            print(f"  {row['band']:14s} {100 * row['psi_power_share']:6.1f}%  capture {row['capture']:6.2f}  coherence {row['coherence']:6.2f}  "
                  f"error ratio {row['error_ratio']:6.2f}  signal loss {row['signal_loss']:7.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2))

    bands = [r["band"] for r in next(iter(report.values()))]
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.2))
    for axis, key, title in zip(ax, ("capture", "error_ratio", "signal_loss"), ("capture of psi (1 = ideal)", "error ratio (<1 better)", "fraction of the TRUE field removed (0 = ideal)")):
        for i, (name, rows) in enumerate(report.items()):
            axis.bar(np.arange(len(bands)) + i * 0.8 / len(report), [r[key] for r in rows], 0.8 / len(report), label=name)
        axis.set_xticks(np.arange(len(bands)) + 0.4)
        axis.set_xticklabels(bands, rotation=60, ha="right")
        axis.set_title(title)
    ax[1].axhline(1, color="k", lw=0.8)
    ax[0].legend()
    plt.tight_layout()
    plt.savefig(args.out.with_suffix(".png"), dpi=130)
    print("\nsaved", args.out.with_suffix(".json"), args.out.with_suffix(".png"))


if __name__ == "__main__":
    main()
