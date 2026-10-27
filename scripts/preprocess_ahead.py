"""AHEAD qsm / r2starmap / t1w -> B0-aligned 1 mm isotropic cache (one npz per subject) for S4c.

Each subject is resampled ONCE with its own scanner-space affine into a grid whose axes are the world axes, so B0 is along z (see
mcno_qsm/ahead.py). Writes <out>/sub-XXXX.npz (float16 chi / weight, uint8 mask) and <out>/summary.json. Resumable (finished
subjects are skipped). The weight is the proxy T1w * exp(-TE R2*), rank-matched to RC1's in-mask `magn` distribution (mean 0.264,
p1-p99 0.13-0.38), because the raw proxy is far wider than what the nets saw in training.
Run (server): nohup .venv/bin/python scripts/preprocess_ahead.py > runs/preprocess_ahead.log 2>&1 &
Only numpy / scipy / nibabel are used: it can also be run with `python -I` after setting PYTHONPATH to src.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio

from mcno_qsm.ahead import find_subjects, preprocess_subject, save_subject


def rc1_weight_quantiles(root, points=1001):
    root = Path(root)
    magnitude = sio.loadmat(root / "magn.mat")["magn"]
    mask = sio.loadmat(root / "msk.mat")["msk"] > 0
    return np.quantile(magnitude[mask].astype(np.float64), np.linspace(0.0, 1.0, points))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("dataAHEAD"))
    parser.add_argument("--out", type=Path, default=Path("dataAHEAD_1mm"))
    parser.add_argument("--te-ms", type=float, default=20.0, help="echo time of the weight proxy exp(-TE R2*), R2* in 1/ms")
    parser.add_argument("--rc1", type=Path, default=Path("dataRC1"), help="RC1 folder: the weight is rank-matched to its in-mask `magn`")
    parser.add_argument("--no-match", action="store_true", help="only scale the weight proxy to the RC1 mean instead of rank-matching")
    parser.add_argument("--limit", type=int, default=None, help="only the first N subjects (smoke test)")
    args = parser.parse_args()
    subjects = find_subjects(args.root)
    if not subjects:
        raise SystemExit(f"no complete qsm/r2starmap/t1w triplets under {args.root}")
    names = list(subjects)[: args.limit]
    reference = None if args.no_match else rc1_weight_quantiles(args.rc1)
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    print(f"{len(subjects)} subjects found, processing {len(names)}", flush=True)
    for name in names:
        target = args.out / f"{name}.npz"
        if target.exists() and name in summary:
            continue
        try:
            arrays, info = preprocess_subject(subjects[name], te_ms=args.te_ms, weight_reference=reference)
        except Exception as exc:                      # keep going; the failure is listed at the end
            summary[name] = {"error": repr(exc)}
            print(f"{name} FAILED: {exc!r}", flush=True)
            summary_path.write_text(json.dumps(summary, indent=1))
            continue
        save_subject(target, arrays, info)
        summary[name] = info
        summary_path.write_text(json.dumps(summary, indent=1))
        print(f"{name} shape {info['out_shape']} mask {info['mask_voxels']} tilt {info['tilt_deg']} deg  chi p0.1/p99.9 "
              f"{info['chi_p0.1']:.3f}/{info['chi_p99.9']:.3f}  weight mean/p1/p99 {info['weight_mean']:.3f}/{info['weight_p1']:.3f}/"
              f"{info['weight_p99']:.3f}", flush=True)
    failed = [n for n, v in summary.items() if "error" in v]
    print(f"preprocess_ahead finished: {len(summary) - len(failed)} ok, {len(failed)} failed {failed}", flush=True)


if __name__ == "__main__":
    main()
