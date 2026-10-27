"""Is the AHEAD NIfTI affine in scanner coordinates (world z = B0)? Prints the evidence; reads headers only (fast).

NIfTI does not store the B0 direction. If qform/sform code is 1 (scanner anatomical) the affine maps voxels to scanner
coordinates, so B0 = world z and its direction in voxel coordinates is inv(R) @ [0, 0, 1]. Code 2 (aligned), 3/4 (Talairach/MNI)
or a rotation that is identical for every subject points to a registered / reoriented volume where this does NOT hold.
Run (project root): .venv/bin/python scripts/inspect_ahead_affine.py --root dataAHEAD
"""

import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np


def describe(path):
    image = nib.load(path)
    header = image.header
    affine = image.affine
    zooms = np.array(header.get_zooms()[:3])
    rotation = affine[:3, :3] / zooms                      # columns = voxel axes in world coordinates
    b0_voxel = np.linalg.inv(rotation) @ np.array([0.0, 0.0, 1.0])
    tilt = np.degrees(np.arccos(np.clip(np.abs(b0_voxel).max() / np.linalg.norm(b0_voxel), -1, 1)))
    return {
        "shape": image.shape, "zooms": zooms.round(4).tolist(), "axcodes": "".join(nib.aff2axcodes(affine)),
        "qform_code": int(header["qform_code"]), "sform_code": int(header["sform_code"]),
        "b0_in_voxel_axes": (b0_voxel / np.linalg.norm(b0_voxel)).round(4).tolist(), "tilt_deg": round(float(tilt), 2),
        "rotation_det": round(float(np.linalg.det(rotation)), 3), "affine": affine.round(4).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("dataAHEAD"))
    parser.add_argument("--subjects", type=int, default=12, help="how many subjects, spread over the whole list")
    args = parser.parse_args()
    files = sorted(args.root.rglob("*mod-qsm*.nii.gz"))
    if not files:
        raise SystemExit(f"no *mod-qsm*.nii.gz under {args.root}")
    chosen = [files[i] for i in np.unique(np.linspace(0, len(files) - 1, min(args.subjects, len(files))).astype(int))]
    print(f"{len(files)} qsm files; header inspection of {len(chosen)}")
    print(f"sidecar json files in the dataset: {len(list(args.root.rglob('*.json')))}")
    rows = []
    for path in chosen:
        info = describe(path)
        rows.append(info)
        print(f"{path.name[:28]:28s} codes q/s {info['qform_code']}/{info['sform_code']}  axes {info['axcodes']}  zooms {info['zooms']}  "
              f"B0 in voxel axes {info['b0_in_voxel_axes']}  tilt {info['tilt_deg']} deg  det {info['rotation_det']}")
    tilts = np.array([r["tilt_deg"] for r in rows])
    b0 = np.array([r["b0_in_voxel_axes"] for r in rows])
    print(f"\ntilt of B0 from the nearest voxel axis (deg): min {tilts.min():.2f}  max {tilts.max():.2f}  std {tilts.std():.2f}")
    print(f"B0 in voxel axes, std across subjects per component: {np.abs(b0).std(0).round(4).tolist()}")
    print("distinct sform codes:", sorted({r['sform_code'] for r in rows}), "| distinct qform codes:", sorted({r['qform_code'] for r in rows}))
    print("\nfull affine of the first subject:")
    print(np.array(rows[0]["affine"]))
    print("\nreading: sform/qform code 1 + a tilt that VARIES between subjects => scanner coordinates (each head has its own pose), B0 = world z, "
          "use b0_in_voxel_axes.\n         code 2/3/4, or an identical affine for every subject (tilt std ~ 0) => the volumes were registered or "
          "reoriented to a common frame; B0 is then NOT recoverable from the affine.")
    Path("runs").mkdir(exist_ok=True)
    Path("runs/ahead_affine.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
