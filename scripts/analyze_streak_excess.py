"""Follow-up of analyze_streak_kspace.py (CLAUDE.md 9ap): chi-error EXCESS over the oracle, in absolute units (% of ||chi||^2),
per Dipole-let segment and on the (|k|, |D|) plane. Reads runs/streak_kspace/maps.npz and recomputes the segment energies."""
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mcno_qsm.dipolelets import SIM2_DIPOLELET, DipoleletTransform  # noqa: E402
from mcno_qsm.physics import dipole_kernel  # noqa: E402
from mcno_qsm.rc2 import load_rc2  # noqa: E402

OUT = Path("runs/streak_kspace")
NAMES = [f"s{j}/b{k}" for j in (1, 2, 3) for k in (0, 1, 2)] + ["coarse"]
data = load_rc2("dataRC2", 7.0)
mask, chi, voxel = data["mask"].astype(np.float32), data["chi"], data["voxel_size"]
shape = chi.shape
z = np.load("runs/chi_fixed25.npz")
chis = {k[4:]: z[k].astype(np.float32) * mask for k in z.files}
tr = DipoleletTransform(shape, voxel, **SIM2_DIPOLELET)
norm2 = float(((chi * mask) ** 2).sum())
def seg_e(x):
    return np.array([float((s[0, 0].abs() ** 2).sum()) for s in tr.analysis(torch.from_numpy(x)[None, None])])
# segments overlap: scale each signal's segment energies so that they sum to the (Parseval) total energy of that signal
res = {}
for m, c in chis.items():
    e = c - chi * mask
    es = seg_e(e.astype(np.float32))
    res[m] = es / es.sum() * float((e.astype(np.float64) ** 2).sum()) / norm2 * 100
lines = ["chi-error energy per segment, % of ||chi||^2 (segment overlap normalised away)",
         f"{'method':22s} " + " ".join(f"{n:>6s}" for n in NAMES) + "   total"]
for m, v in res.items():
    lines.append(f"{m:22s} " + " ".join(f"{x:6.2f}" for x in v) + f"  {v.sum():6.2f}")
for m in ("dirty", "noise_clean"):
    d = res[m] - res["oracle"]
    lines.append(f"{'EXCESS ' + m:22s} " + " ".join(f"{x:6.2f}" for x in d) + f"  {d.sum():6.2f}")
    lines.append(f"{'  share of excess %':22s} " + " ".join(f"{100 * x / d.sum():6.1f}" for x in d))
cut = 100 * (1 - res["noise_clean"].sum() and 0)  # placeholder to keep flake quiet
# excess on the (|k|, |D|) plane
m = np.load(OUT / "maps.npz")
ke, de = m["k_edges"], m["d_edges"]
tot_nc = m["e_nc"].sum()
for name, key in (("dirty", "e_dirty"), ("noise_clean", "e_nc")):
    ex = (m[key] - m["e_oracle"]) / m["chi"].sum() * 100 * (res[name if name != "noise_clean" else "noise_clean"].sum() / 100 / (m[key].sum() / m["chi"].sum()))
    lines.append(f"\nEXCESS chi-error energy {name} - oracle on the (|k| rows, |D| cols) plane, % of ||chi||^2 x100 (x0.01 %)")
    lines.append(" " * 12 + " ".join(f"{e:6.3f}" for e in de[1:]) + "   row sum")
    for i in range(len(ke) - 1):
        lines.append(f"{ke[i]:5.3f}-{ke[i + 1]:5.3f} " + " ".join(f"{100 * x:6.1f}" for x in ex[i]) + f"   {ex[i].sum():6.3f}")
    lines.append(f"{'col sum':12s} " + " ".join(f"{x:6.3f}" for x in ex.sum(0)) + f"   {ex.sum():6.3f}")
# coarse remainder: split by |D|
(OUT / "excess.txt").write_text("\n".join(lines))
print("\n".join(lines))
