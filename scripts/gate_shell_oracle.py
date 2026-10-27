"""G1 (CLAUDE.md 9ap): headroom of the calcification shell. eNDI (25 fixed iterations, as diagnose_segments.invert) of
(a) noise_clean cleaned field, (b) the same with the TRUE field put back inside the <=3 mm shell of the 37 calcification voxels,
(c) the same inside a 6 mm shell, (d) dirty with the true shell, (e) oracle. Sim2 used for DIAGNOSIS only (nothing is trained)."""
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_segments import invert  # noqa: E402
from mcno_qsm.rc2 import load_rc2  # noqa: E402
from mcno_qsm.scoring import forward_field  # noqa: E402

data = load_rc2("dataRC2", 7.0)
mask, chi, w, voxel = data["mask"].astype(np.float32), data["chi"], data["weight"], data["voxel_size"]
b = forward_field(chi, mask, voxel)
dirty = data["field"] * mask
clean = nib.load("runs/s4b_noise_clean/best_clean.nii.gz").get_fdata(dtype=np.float32) * mask
calc = chi < -0.5
dist = ndimage.distance_transform_edt(~calc, sampling=voxel)
roi = ndimage.binary_dilation(calc, iterations=8)
runs = {"oracle": b, "noise_clean": clean}
for r in (3.0, 6.0):
    sh = (dist <= r) & (mask > 0)
    runs[f"nc + true shell<={r:g}mm ({int(sh.sum())} vox)"] = np.where(sh, b, clean)
    runs[f"dirty + true shell<={r:g}mm"] = np.where(sh, b, dirty)
runs["dirty"] = dirty
for name, f in runs.items():
    t = time.time()
    r = invert(f, mask, w, chi, 25)
    print(f"{name:42s} eNDI NRMSE@25 = {r:6.2f}   ({time.time() - t:.0f}s)", flush=True)
